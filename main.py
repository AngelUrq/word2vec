import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open('data.txt', encoding='ISO-8859-1') as file:
    full_text = file.read()

tokenizer = get_tokenizer('basic_english')

vocab = build_vocab_from_iterator(full_text, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(f'Len of vocab {len(vocab)}')

vocab(['asi', ',', 'la', 'mañana'])
