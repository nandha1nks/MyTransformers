from nltk.tokenize import word_tokenize
from tqdm import tqdm


def get_vocab(sentences, language="english"):
    max_seq_len = 0
    vocabs = {
        "<START>": 0,
        "<END>": 1,
        "<UNK>": 2,
    }
    idx = 3
    for sent in tqdm(sentences):
        te = word_tokenize(sent, language=language)
        max_seq_len = max(max_seq_len, len(te))
        for t in te:
            if t not in vocabs:
                vocabs[t] = idx
                idx += 1
    return vocabs, max_seq_len


class MyTokenizer:
    def __init__(self, vocabs: dict, max_seq_len, pad_token="<END>", pad_right=True, language="english"):
        self.vocabs = vocabs
        self.idx_to_token = {y:x for x, y in self.vocabs.items()}
        self.max_seq_len = max_seq_len
        self.pad_token = vocabs.get("<END>", 1)
        self.pad_right = pad_right
        self.language = language
        self.unk_token = self.vocabs.get("<UNK>", 2)

    def tokenize(self, sent):
        tokens = word_tokenize(sent, language=self.language)
        tokens = tokens[:self.max_seq_len-2]
        tokens = [0] + [self.vocabs.get(x, self.unk_token) for x in tokens] + [1]
        return tokens

    def detokenize(self, tokens):
        return " ".join(self.idx_to_token[t] for t in tokens)
