import torch
from torchtext.datasets import IMDB
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


class IMDBDataLoader:
    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 2):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.text_pipeline = None
        self.label_pipeline = None

    def load_data(self, val_size = 0.1):
        # Define tokenizer
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        # Load IMDB dataset
        train_iter, test_iter = IMDB(split=('train', 'test'), root='./data')

        train_proportion = 1 - val_size

        # Convert train_iter to list for splitting
        train_data = list(train_iter)
        train_size = int(train_proportion * len(train_data))
        train_data, val_data = train_data[:train_size], train_data[train_size:]

        # Build vocabulary using GloVe embeddings
        counter = Counter()
        for (label, line) in train_data:
            counter.update(tokenizer(line))

        # Initialize GloVe embeddings
        glove = GloVe(name='6B', dim=100)

        # Create a custom vocabulary that maps tokens to indices
        vocab = {token: idx for idx, token in enumerate(glove.stoi.keys())}
        vocab['<unk>'] = len(vocab)  # Add unknown token

        # Define text and label pipelines
        self.text_pipeline = lambda x: [vocab.get(token, vocab['<unk>']) for token in tokenizer(x)]
        self.label_pipeline = lambda x: 1 if x == 'pos' else 0

        # Create iterators for train, validation and test
        train_iter = iter(train_data)
        val_iter = iter(val_data)
        test_iter = iter(test_iter)
    
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collate_batch)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_batch)
        test_loader = DataLoader(test_iter, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_batch)

        return train_loader, val_loader, test_loader

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.float)
            text_list.append(processed_text)
        text_list = pad_sequence(text_list, batch_first=True)
        label_list = torch.tensor(label_list, dtype=torch.float)
        return text_list, label_list