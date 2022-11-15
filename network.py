import torch
from torch import nn
import torch.nn.functional as F
from torch import einsum
from torch.nn.parameter import Parameter
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import numpy as np
import math
from utils import shift_output_sequence, create_shifted_output_mask

def init_to_zeros(m):
    '''
    Initializes LayerNorm and Linear layers to zero
    
    args:
        m - model to initialize
    '''
    if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Linear): 
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

def initialize_model(model, pretrained, n=12):
    '''
    Initializes all layers in ViT to weights from pretrained ViT

    args:
        model - model to initialize
        pretrained - pretrained model
        n - number of transformer layers to be initialized
    '''
    pretrained_transformer = pretrained.transformer.blocks
    for i, child in pretrained_transformer.named_children():
        i = int(i)
        if i > n: break

        model.transformer.layers[i].attn.load_state_dict(child.state_dict(), strict=False)
        model.transformer.layers[i].mlp.load_state_dict(child.pwff.state_dict(), strict=False)
        model.transformer.layers[i].mlp.load_state_dict(child.state_dict(), strict=False)
    return model

class PositionalEncodingComponent(nn.Module):
    '''
    Class to encode positional information to tokens.
    For future, I want that this class to work even for sequences longer than 5000

    From: https://github.com/akashe/Multimodal-action-recognition
    '''

    def __init__(self, hid_dim, dropout=0.2, max_len=5000):
        super().__init__()

        assert hid_dim % 2 == 0  # If not, it will result error in allocation to positional_encodings[:,1::2] later

        self.dropout = nn.Dropout(dropout)

        self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, hid_dim), requires_grad=False)
        # Positional Embeddings : [1,max_len,hid_dim]

        pos = torch.arange(0, max_len).unsqueeze(1)  # pos : [max_len,1]
        div_term = torch.exp(-torch.arange(0, hid_dim, 2) * math.log(
            10000.0) / hid_dim)  # Calculating value of 1/(10000^(2i/hid_dim)) in log space and then exponentiating it
        # div_term: [hid_dim//2]

        self.positional_encodings[:, :, 0::2] = torch.sin(pos * div_term)  # pos*div_term [max_len,hid_dim//2]
        self.positional_encodings[:, :, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        # TODO: update this for very long sequences
        x = x + self.positional_encodings[:, :x.size(1)].detach()
        return self.dropout(x)

# To do: Fix this
class SPT(nn.Module):
    '''
    Few shot learning technique. Special way to tokenzie inputs. I don't understand it rn
    but it's here to mess around with
    '''
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

# To do: Fix this
class LSA(nn.Module):
    '''
    Another few shot learning technique from the same paper. Again, idk what's going on
    but we can mess around later
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.proj_q = nn.Linear(dim, dim, bias=True)
        self.proj_k = nn.Linear(dim, dim, bias=True)
        self.proj_v = nn.Linear(dim, dim, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def split_last(self, x, shape):
        "split the last dimension to given shape"
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)

    def forward(self, x):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        q, k, v = (self.split_last(x, (self.heads, -1)).transpose(1, 2) for x in [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FSAttention(nn.Module):
    """Factorized Self-Attention"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.proj_q = nn.Linear(dim, dim, bias=True)
        self.proj_k = nn.Linear(dim, dim, bias=True)
        self.proj_v = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def split_last(self, x, shape):
        "split the last dimension to given shape"
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)

    def forward(self, x):

        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        q, k, v = (self.split_last(x, (self.heads, -1)).transpose(1, 2) for x in [q, k, v])

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.drop(out)
        return out

class AttentionBlock(nn.Module):
    '''
    
    '''
    def __init__(self, dim, n_heads, dropout=0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = FSAttention(dim, heads=n_heads, dropout=dropout)
        # self.attn = LSA(dim, heads=n_heads, dropout=dropout)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0, activation='relu'):
        super().__init__()

        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim*4)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.fc2 = nn.Linear(dim*4, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim=512, n_heads=4, n_layers=4, dropout=0):
        super().__init__()
        self.dim = dim
        
        attn = AttentionBlock(dim, n_heads=n_heads, dropout=dropout)
        mlp = FeedForward(dim, dropout=dropout)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            temp = nn.ModuleList([])
            temp.add_module('attn', attn)
            temp.add_module('mlp', mlp)
            self.layers.append(temp)

    def forward(self, z):
        # z = torch.flatten(z, start_dim=1, end_dim=2) 
        for attn, mlp in self.layers:
            z = attn(z) + z
            z = mlp(z) + z

        return z

class TransformerDecoder(nn.Module):
    def __init__(self, dim=512, n_heads=4, n_layers=4, dropout=0):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, dropout=dropout)

        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            temp = nn.ModuleList([])
            temp.add_module('masked_attn', self.attn1)
            temp.add_module('norm', self.norm)
            temp.add_module('attn', self.attn2)
            temp.add_module('ff', self.ff)
            self.layers.append(temp)

    def forward(self, in_seq, out_seq, padding_mask, shifted_output_mask):
        '''
        Args:
            in_seq: Input sequence of size [batch_size, in_seq_len, dim]
            out_seq: Output sequence of size [batch_sze, out_seq_len, dim]
            padding_mask: Padding mask of size [batch_szie, in_seq_len, in_seq_len]
                None if no mask is used
            shifted_output_mask: Shifted output mask of size [batch_size, out_seq_len, in_seq_len]

        Returns:
            Computed output of size [batch_size, out_seq_len, dim]
        '''

        for masked_attn, norm, attn, ff in self.layers:
            # Masked Attention
            out_seq = norm(
                self.dropout(
                    masked_attn(out_seq, out_seq, out_seq, attn_mask=shifted_output_mask, need_weights=False)[0]
                ) + out_seq
            )

            # Cross Attention
            out_seq = norm(
                self.dropout(
                    attn(out_seq, in_seq, in_seq, attn_mask=padding_mask, need_weights=False)[0]
                ) + out_seq
            )

            # Feed forward
            out_seq = norm(
                self.dropout(
                    ff(out_seq)
                ) + out_seq
            )

        return out_seq
        

class Model(nn.Module):
    def __init__(self, 
                i_dim=(256, 512),
                p_dim=(64, 64),
                dim=512,
                n_heads=4,
                n_encoder_layers=6,
                n_decoder_layers=6,
                dropout=0,
                vocab_len=1000,
                device='cpu'):
        
        super().__init__()

        W, H = i_dim, i_dim # temp solution
        w, h = p_dim, p_dim
        nw, nh = W // w, H // h
        self.dim = dim

        proj_dim = 3 * w * h
        self.img_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=h, pw=w),
            nn.Linear(proj_dim, dim)
        )
        # self.embedding = SPT(dim=dim, patch_size=w)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.img_pos_embedding = nn.Parameter(torch.randn(1, nh * nw, dim))
        self.mlp_head = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(dim, 384),
                                    nn.BatchNorm1d(384),
                                    nn.ReLU(),
                                    nn.Linear(384, 1),
                                    nn.Sigmoid())

        self.encoder = TransformerEncoder(dim=dim, n_heads=n_heads, n_layers=n_encoder_layers, dropout=dropout)
        self.decoder = TransformerDecoder(dim=dim, n_heads=n_heads, n_layers=n_decoder_layers, dropout=dropout)
        self.fc = nn.Linear(dim, vocab_len)
        self.softmax = nn.Softmax()

        self.word_embedding = nn.Embedding(vocab_len, dim)
        self.word_pos_embedding = PositionalEncodingComponent(dim, dropout, h*w) # Third param = embedding length

    def forward(self, img, caption):
        #Expects: [b, c, h, w]
        embedded_img = self.img_embedding(img)
        embedded_img += self.img_pos_embedding

        encoded_img = self.encoder(embedded_img)

        # Prepare caption
        caption = self.word_embedding(caption)
        caption = self.word_pos_embedding(caption)

        caption = shift_output_sequence(caption) # I have no idea why we do this
        output_mask = create_shifted_output_mask(caption) # I also do not understand this

        decoded = self.decoder(encoded_img, caption, padding_mask=None, shifted_output_mask=output_mask)

        # Linear output

        return self.softmax(self.fc(decoded))