import torch
from torch import nn
import torch.nn.functional as F
from torch import einsum
from torch.nn.parameter import Parameter
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import numpy as np


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
    def __init__(self, dim, dropout=0):
        super().__init__()

        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim*4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(dim*4, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.gelu(x)
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
        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential([
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim*4, dim)
        ])

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
            x = masked_attn(out_seq, out_seq, out_seq, attn_mask=shifted_output_mask)
            x = self.dropout(x) + out_seq
            x = norm(x)

            # Regular Attention
            y = attn(x, in_seq, in_seq, mask=padding_mask)
            y = self.dropout(y) + x
            y = norm(y)

            # Feed forward
            z = ff(y)
            z = self.dropout(z) + y
            z = norm(z)
        

class Model(nn.Module):
    def __init__(self, 
                i_dim=(256, 512),
                p_dim=(64, 64),
                dim=512,
                n_heads=4,
                n_transformer_layers=12,
                dropout=0,
                device='cpu'):
        
        super().__init__()

        W, H = i_dim, i_dim # temp solution
        w, h = p_dim, i_dim
        nw, nh = W // w, H // h
        self.dim = dim

        proj_dim = 3 * w * h
        self.embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=h, pw=w),
            nn.Linear(proj_dim, dim)
        )
        # self.embedding = SPT(dim=dim, patch_size=w)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, nh * nw, dim))
        self.mlp_head = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(dim, 384),
                                    nn.BatchNorm1d(384),
                                    nn.ReLU(),
                                    nn.Linear(384, 1),
                                    nn.Sigmoid())

        self.transformer = TransformerEncoder(dim=dim, n_heads=n_heads, n_layers=n_transformer_layers, dropout=dropout)

    def forward(self, x):
        #Expects: [b, c, h, w]
        z = self.embedding(x)
        z += self.pos_embedding
        
        b = z.size()[0]
        cls_to_batch = self.cls_token.expand([b, -1, -1])
        z = torch.cat((cls_to_batch, z), dim=1)

        encoded = self.transformer(z)[:, 0, :] #Take only the first index at each batch (cls_token)
        # logits = self.mlp_head(encoded)

        return encoded