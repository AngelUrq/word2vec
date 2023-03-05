import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import re

from dataset import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class WordsDataset(torch.utils.data.Dataset):
    
    def __init__(self, context_window, context_first=True):
        self.dataset = load_dataset('large_spanish_corpus', name='all_wikis', split='train')[:100000]
        self.tokenizer = get_tokenizer('basic_english')

        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.dataset['text']), specials=['<unk>'], min_freq=1)
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.context_window = context_window

        print(f'Len of vocab {len(self.vocab)}')

        self.X = []
        self.y = []

        for document in self.dataset['text']:
          document = re.sub('[^A-Za-z0-9áéíóúñÁÉÍÓÚÑ]+', ' ', document)
          tokens = self.tokenizer(document)

          for i, word in enumerate(tokens):
              center = word

              if i < context_window:
                  continue
              elif i > len(tokens) - context_window - 1:
                  continue
              else:
                  context = tokens[i - context_window:i] + tokens[i + 1:i + context_window + 1]

              if context_first:
                self.X.append(self.vocab(context))
                self.y.append(self.vocab([center]))
              else:
                self.X.append(self.vocab([center]))
                self.y.append(self.vocab(context))

        if context_first:
          self.X = torch.tensor(self.X).long()
          self.y = torch.tensor(self.y).squeeze().long()
        else:
          self.X = torch.tensor(self.X).squeeze().long()
          self.y = torch.tensor(self.y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    