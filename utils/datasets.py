import json

from torch.utils.data import Dataset
import torch


class TranslationDataset(Dataset):
    def __init__(self, enc_sentences, dec_sentences, enc_tokenizer, dec_tokenizer):
        self.enc_sentences = enc_sentences
        self.enc_tokenizer = enc_tokenizer
        self.dec_sentences = dec_sentences
        self.dec_tokenizer = dec_tokenizer

    def __len__(self):
        return min(len(self.enc_sentences), len(self.dec_sentences))

    def __getitem__(self, index):
        src = torch.tensor(self.enc_tokenizer.tokenize(self.enc_sentences[index]))
        tar = torch.tensor(self.dec_tokenizer.tokenize(self.dec_sentences[index]))
        return src, tar


class SentenceCompletionDataset(Dataset):
    def __init__(self, enc_sentences, enc_tokenizer):
        self.enc_sentences = enc_sentences
        self.enc_tokenizer = enc_tokenizer

    def __len__(self):
        return len(self.enc_sentences)

    def __getitem__(self, index):
        src = torch.tensor(self.enc_tokenizer.tokenize(self.enc_sentences[index]))
        return src


class NERDataset(Dataset):
    def __init__(self, json_file):
        self.data = json.load(open(json_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]["sentence"], \
            json.dumps(self.data[item]["ner"])
