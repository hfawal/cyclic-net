import torch
from torchtext.datasets import IMDB
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
import pickle
import os


class IMDBDataLoader:
    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 2):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.vocab = None
        self.embeddings = None
        self.reverse_vocab = None
        self.cache_dir = './cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def text_pipeline(self, x):
        return [self.vocab.get(token, self.vocab['<unk>']) for token in self.tokenizer(x)]

    def label_pipeline(self, x):
        return 1 if x == 'pos' else 0

    def load_data(self, val_size = 0.1):
        # Define tokenizer
        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        # Load IMDB dataset
        train_iter, test_iter = IMDB(split=('train', 'test'), root='./data')

        train_proportion = 1 - val_size

        # Convert train_iter to list for splitting
        train_data = list(train_iter)
        train_size = int(train_proportion * len(train_data))
        train_data, val_data = train_data[:train_size], train_data[train_size:]

        # Try to load cached vocabulary and embeddings
        cache_path = os.path.join(self.cache_dir, 'vocab_embeddings.pkl')
        if os.path.exists(cache_path):
            print("Loading cached vocabulary and embeddings...")
            with open(cache_path, 'rb') as f:
                self.vocab, self.embeddings = pickle.load(f)
        else:
            print("Generating new vocabulary and embeddings...")
            # Build vocabulary using GloVe embeddings
            counter = Counter()
            for (label, line) in train_data:
                counter.update(self.tokenizer(line))

            # Initialize GloVe embeddings
            glove = GloVe(name='6B', dim=100)
            
            # Create a custom vocabulary that maps tokens to indices
            self.vocab = {token: idx for idx, token in enumerate(glove.stoi.keys())}
            self.vocab['<unk>'] = len(self.vocab)  # Add unknown token

            # Create <unk> vector as mean of all GloVe vectors
            unk_vector = torch.mean(glove.vectors, dim=0)
            
            # Concatenate the <unk> vector to the embeddings
            self.embeddings = torch.cat([glove.vectors, unk_vector.unsqueeze(0)], dim=0)

            # Cache the vocabulary and embeddings
            print("Caching vocabulary and embeddings...")
            with open(cache_path, 'wb') as f:
                pickle.dump((self.vocab, self.embeddings), f)

        # Create reverse vocabulary mapping
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collate_batch)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_batch)
        test_loader = DataLoader(test_iter, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_batch)

        return train_loader, val_loader, test_loader

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            # Get token indices
            token_indices = self.text_pipeline(_text)
            # Convert indices to embeddings
            embeddings = self.embeddings[token_indices]
            text_list.append(embeddings)
        # Pad the sequences of embeddings
        text_list = pad_sequence(text_list, batch_first=True)
        label_list = torch.tensor(label_list, dtype=torch.float)
        return text_list, label_list

    def decode_embeddings(self, embeddings):
        """Convert embeddings back to tokens using cosine similarity"""
        # Normalize embeddings for cosine similarity
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        normalized_vocab = torch.nn.functional.normalize(self.embeddings, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(normalized_embeddings, normalized_vocab.T)
        
        # Get the most similar token for each embedding
        token_indices = torch.argmax(similarity, dim=-1)
        
        # Convert indices to tokens
        tokens = [self.reverse_vocab[idx.item()] for idx in token_indices]
        return tokens

    def tokens_to_text(self, tokens):
        """Convert a list of tokens back to text"""
        return ' '.join(tokens)

    def decode_embeddings_to_text(self, embeddings):
        """Convert embeddings directly to text"""
        tokens = self.decode_embeddings(embeddings)
        return self.tokens_to_text(tokens)

    def get_last_nonzero_index(self, data):
        """Find the index of the last nonzero element in a tensor.
        
        Args:
            data (torch.Tensor): Input tensor of any shape
            
        Returns:
            torch.Tensor: Indices of last nonzero elements along the last dimension
        """
        # Convert to boolean mask of nonzeros
        nonzero_mask = (data != 0)
        
        # Get the last index where mask is True along the last dimension
        last_nonzero = torch.max(torch.where(nonzero_mask, torch.arange(data.size(-1), device=data.device), torch.tensor(0, device=data.device)), dim=-1)[0]
        
        return last_nonzero
