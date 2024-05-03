import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model.transformer import Transformer
from utils.tokenizer import MyTokenizer, get_vocab
from utils.datasets import TranslationDataset

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE {device}")

SRC_LANGUAGE = "english"
TGT_LANGUAGE = "german"
TRAIN_SRC_SENTENCES_FILE = "../data/eng-ger/sample.en"
TRAIN_TGT_SENTENCES_FILE = "../data/eng-ger/sample.de"
VAL_SRC_SENTENCES_FILE = "../data/eng-ger/s.en"
VAL_TGT_SENTENCES_FILE = "../data/eng-ger/s.de"

MAX_STEPS = 100000
d_model = 512
warmup_steps = 4000
optimizer_lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9
grad_clip_threshold = 1.0

train_src_sents = open(TRAIN_SRC_SENTENCES_FILE).readlines()
train_tgt_sents = open(TRAIN_TGT_SENTENCES_FILE).readlines()
val_src_sents = open(VAL_SRC_SENTENCES_FILE).readlines()[:1024]
val_tgt_sents = open(VAL_TGT_SENTENCES_FILE).readlines()[:1024]
print("Loaded Sentences")

src_vocabs, src_max_seq_len = get_vocab(train_src_sents, SRC_LANGUAGE)
print(
    f"GOT VOCAB For SRC: MAX_SEQ_LEN {src_max_seq_len} Len {len(src_vocabs)}")
tgt_vocabs, tgt_max_seq_len = get_vocab(train_src_sents, TGT_LANGUAGE)
print(
    f"GOT VOCAB For TGT: MAX_SEQ_LEN {tgt_max_seq_len} Len {len(tgt_vocabs)}")

MAX_SEQ_LEN = min(256, max(tgt_max_seq_len, src_max_seq_len))

src_tokenizer = MyTokenizer(src_vocabs, MAX_SEQ_LEN, pad_token="<END>", pad_right=True, language=SRC_LANGUAGE)
tgt_tokenizer = MyTokenizer(tgt_vocabs, MAX_SEQ_LEN, pad_token="<END>", pad_right=False, language=TGT_LANGUAGE)

train_dataset = TranslationDataset(train_src_sents, train_tgt_sents, src_tokenizer, tgt_tokenizer)
val_dataset = TranslationDataset(val_src_sents, val_tgt_sents, src_tokenizer, tgt_tokenizer)


def get_tensor_and_mask(token_lists, pad_token):
    max_len = max(len(sentence) for sentence in token_lists)
    max_len = min(MAX_SEQ_LEN, max_len)
    token_lists = [s[:max_len] for s in token_lists]

    # Create a tensor to hold the padded sequences with appropriate dtype
    token_tensor = torch.full((len(token_lists), max_len), pad_token,
                              dtype=torch.long, )  # Adjust dtype if tokens are strings

    # Create a tensor for the attention mask
    attention_mask = torch.zeros((len(token_lists), max_len), dtype=torch.long)

    # Pad the sequences and create the attention mask efficiently
    for i, sentence in enumerate(token_lists):
        token_tensor[i, :len(sentence)] = sentence
        attention_mask[i, :len(sentence)] = 1

    return token_tensor.to(device), attention_mask.to(device)


def collate_fn(batch):
    srcs = [x[0] for x in batch]
    tgts = [x[1] for x in batch]

    return get_tensor_and_mask(srcs, src_tokenizer.pad_token) + \
        get_tensor_and_mask(tgts, tgt_tokenizer.pad_token)


model = Transformer(len(src_vocabs), len(tgt_vocabs),
                    MAX_SEQ_LEN, 512, 1024, 8, 6, device)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=optimizer_lr,
    betas=betas,
    eps=eps
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


def lr_schedule(step_num_):
    return d_model ** -0.5 * min(step_num_ ** -0.5, step_num_ * warmup_steps ** -1.5)


print("MAX_SEQ_LEN", MAX_SEQ_LEN)
step_num = 0
epoch = 0
while step_num < MAX_STEPS:
    model.train()
    train_loss = 0

    for idx, d in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        src, src_attn, tgt, tgt_attn = d
        pred = model(src, tgt[:, :-1], src_attn, tgt_attn[:, :-1])

        flattened_mask = tgt_attn[:, :-1].reshape(-1)
        loss = criterion(pred.view(-1, len(tgt_vocabs))[flattened_mask == 1],
                         tgt[:, 1:].reshape(-1)[flattened_mask == 1])
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_threshold)

        current_lr = lr_schedule(step_num + 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.step()
        train_loss += loss.item()
        step_num += 1

        if (idx + 1) % 1 == 0:
            print(f"Train {epoch} iter {(idx + 1) / 1} train loss {train_loss / (idx + 1)}")

    average_train_loss = train_loss / len(train_loader)
    # torch.save(model.state_dict(), f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), f"epoch.pth")

    model.eval()
    all_references = []
    all_candidates = []

    with torch.no_grad():
        for idx, d in tqdm(enumerate(val_loader), total=len(val_loader)):
            src, src_attn, tgt, tgt_attn = d

            pred, pred_attn = model.predict(src, src_attn)
            references = pred.cpu().numpy().tolist()
            candidates = tgt.cpu().detach().tolist()

            all_references.extend(references)
            all_candidates.extend(candidates)

    bleu_score = corpus_bleu([[r] for r in all_references], [c for c in all_candidates])

    print(f"Epoch [{epoch + 1}] => "
          # f"Train Loss: {average_train_loss:.4f}, "
          f"Validation BLEU: {bleu_score:.4f}")

    epoch += 1
