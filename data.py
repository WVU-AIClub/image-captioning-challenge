from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import load_dataset
import spacy
from collections import Counter
import re
import unicodedata


class BloomData(Dataset):
    def __init__(self, img_size, lang):
        assert lang in ['tha', 'kir', 'hau'], f'Expected selected language to be "tha", "kir", or "hau". Got {lang}'

        # preprocess = T.Compose([
        #     T.CenterCrop(),
        #     T.Resize(img_size)
        # ])
        self.lang = lang
        self.dataset = load_dataset("sil-ai/bloom-captioning", self.lang, 
                       use_auth_token=True, download_mode='force_redownload')
        train_captions = [self.dataset['train'][i]['caption'] for i in range(len(self.dataset['train']))]
        self.vocab = self.get_vocab(strings=train_captions)
        
        # Load images here
        
    def __len__(self):
        return

    def __getitem__(self, idx):

        return

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

                tokenized.append(self.vocab[w])
        else:
            caption.replace('\n', ' ').split(' ')
            for word in caption:
                w = unicodedata.normalize('NFKD', word.lower()).strip()
                if w in [' ', '', '\n']: continue

                tokenized.append(self.vocab[w])
        return tokenized

    def get_vocab(self, strings):
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
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

        vocab.update({token: i + len(vocab) for i, token in enumerate(words)})

        # To do: Reverse lookup
        return vocab
