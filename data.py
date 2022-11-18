from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import load_dataset
import spacy
from collections import Counter
import re
import unicodedata
import torch
from PIL import Image
import os
from tqdm import tqdm

class BloomData(Dataset):
    def __init__(self, root, lang, img_size=224, seq_len=196, priority='speed', split='train'):
        assert lang in ['tha', 'kir', 'hau'], f'Expected selected language to be "tha", "kir", or "hau". Got {lang}'

        self.root = root
        self.lang = lang
        self.split = split
        self.priority = priority
        self.seq_len = seq_len

        # Label Loading
        self.dataset = load_dataset("sil-ai/bloom-captioning", self.lang, 
                       use_auth_token=True)
        train_captions = [self.dataset['train'][i]['caption'] for i in range(len(self.dataset['train']))]
        valid_captions = [self.dataset['validation'][i]['caption'] for i in range(len(self.dataset['validation']))]
        self.word2ind, self.ind2word = self.get_vocab(strings=(train_captions + valid_captions))
        self.vocab_len = len(self.word2ind)
        
        # Image loading
        self.preprocess = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

        self.data = {}
        for item in tqdm(self.dataset[split], f'Loading data from {lang} -> {split}'):
            _id = item['image_id']
            filename = f'{_id}.jpg'
            img = os.path.join(root, lang, split, filename)

            if self.priority == 'speed':
                img = Image.open(os.path.join(root, lang, split, filename)).convert('RGB')
                img = self.preprocess(img)
            self.data[_id] = {
                'img': img,
                'caption': item['caption']
            }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _id = list(self.data)[idx]
        img = self.data[_id]['img']
        if self.priority == 'space':
            img = Image.open(os.path.join(self.root, self.lang, self.split, img)).convert('RGB')
            img = self.preprocess(img)
        img = img[:3] # Take first 3 channels from img (handles RGBA images)

        caption = self.tokenize(self.data[_id]['caption'])

        # Return img, caption
        return img, caption

    def load_images(self):
        return

    def tokenize(self, caption):
        tokenized = []
        if self.lang in ['tha', 'kir']:
            spacy_lang = 'th' if self.lang == 'tha' else 'ky'
            nlp = spacy.blank(spacy_lang)
            doc = nlp(caption)
            for word in doc:
                w = unicodedata.normalize('NFKD', word.text.lower()).strip()
                if w in [' ', '', '\n']: continue

                tokenized.append(self.word2ind[w])
        else:
            caption.replace('\n', ' ').split(' ')
            for word in caption:
                w = unicodedata.normalize('NFKD', word.lower()).strip()
                if w in [' ', '', '\n']: continue

                tokenized.append(self.word2ind[w])

        # Append <eos> to caption
        tokenized.append(2)

        # Pad to seq_len
        tokenized += [0]*(self.seq_len - len(tokenized))
        return torch.IntTensor(tokenized)

    def untokenize(self, seq):
        caption = []
        for ind in seq:
            word = self.ind2word[ind]
            if word == '<eos>': break
            caption.append(word)

        return caption

    def get_vocab(self, strings):
        word2ind = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        if self.lang in ['tha', 'kir']:
            spacy_lang = 'th' if self.lang == 'tha' else 'ky'
            nlp = spacy.blank(spacy_lang)
            words = Counter()
            for doc in nlp.pipe(strings):
                for word in doc:
                    w = unicodedata.normalize('NFKD', word.text.lower()).strip()
                    if w in [' ', '', '\n']: continue
                    words[w] += 1
        else:
            # Spacy does not support huasa, so we improvise
            words = set()
            for s in strings:
                # Remove punctuation
                s = re.sub(r'[^\w\s]', '', s)

                # Split on space, lowercase, and add to set
                s = s.replace('\n', ' ').split(' ')
                for word in s:
                    w = unicodedata.normalize('NFKD', word.lower()).strip()
                    if w in [' ', '', '\n']: continue
                    words.add(w)

        word2ind.update({token: i + len(word2ind) for i, token in enumerate(words)})

        # To do: Reverse lookup
        ind2word = {index: caption for caption, index in word2ind.items()}
        return word2ind, ind2word
