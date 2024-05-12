from nltk.translate.bleu_score import corpus_bleu
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model.gpt import GPT
from utils.tokenizer import MyTokenizer, get_vocab
from utils.datasets import SentenceCompletionDataset

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE {device}")

LANGUAGE = "english"
TRAIN_SENTENCES_FILE = "../data/eng-ger/sample.en"
VAL_SENTENCES_FILE = "../data/eng-ger/s.en"

MAX_STEPS = 100000
d_model = 512  # Replace with your model's dimensionality
warmup_steps = 4000
optimizer_lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9

train_sents = open(TRAIN_SENTENCES_FILE).readlines()
val_sents = open(VAL_SENTENCES_FILE).readlines()
print("Loaded Train SRC Sents")

vocabs, max_seq_len = get_vocab(train_sents, "english")
print(
    f"GOT VOCAB For SRC: MAX_SEQ_LEN {max_seq_len} Len {len(vocabs)}")

MAX_SEQ_LEN = min(256, max_seq_len)

tokenizer = MyTokenizer(vocabs, MAX_SEQ_LEN, pad_token="<END>", pad_right=True, language=LANGUAGE)

train_dataset = SentenceCompletionDataset(train_sents, tokenizer)
val_dataset = SentenceCompletionDataset(val_sents, tokenizer)


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
    return get_tensor_and_mask(batch, tokenizer.pad_token)


model = GPT(
    len(vocabs),
    MAX_SEQ_LEN,
    512,
    1024,
    8,
    6,
    device
)
model = model.to(device)
learning_rate = 0.0001
grad_clip_threshold = 1.0

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=optimizer_lr,
    betas=betas,
    eps=eps
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


def lr_schedule(step_num_):
    return d_model ** -0.5 * min(step_num_ ** -0.5, step_num_ * warmup_steps ** -1.5)


model.load_state_dict(torch.load("sample_epoch.pth"))
print("MAX_SEQ_LEN", MAX_SEQ_LEN)
step_num = 0
epoch = 0
while step_num < 5000:
    model.train()
    train_loss = 0

    for idx, d in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        tgt, tgt_attn = d
        pred = model(tgt[:, :-1], tgt_attn[:, :-1])

        flattened_mask = tgt_attn[:, :-1].reshape(-1)
        loss = criterion(pred.view(-1, len(vocabs))[flattened_mask == 1],
                         tgt[:, 1:].reshape(-1)[flattened_mask == 1])
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_threshold)

        optimizer.step()
        train_loss += loss.item()
        step_num += 1

        if (idx + 1) % 100 == 0:
            print(f"Train {idx} iter {(idx + 1) // 100} train loss {train_loss / (idx + 1)}")

    average_train_loss = train_loss / len(train_loader)
    print(average_train_loss)
    torch.save(model.state_dict(), f"sample_epoch.pth")

    all_references = []
    all_candidates = []

    with torch.no_grad():
        for idx, d in tqdm(enumerate(val_loader), total=len(val_loader)):
            src, attn = d

            pred, pred_attn = model.predict(src[:, :6], attn[:, :6])
            references = pred.cpu().numpy().tolist()
            candidates = src.cpu().detach().tolist()

            all_references.extend(references)
            all_candidates.extend(candidates)

    bleu_score = corpus_bleu([[r] for r in all_references], [c for c in all_candidates])

    print(f"Epoch [{epoch + 1}] => "
          # f"Train Loss: {average_train_loss:.4f}, "
          f"Validation BLEU: {bleu_score:.4f}")

    epoch += 1


# model.load_state_dict(torch.load("sample_epoch.pth"))
#
# model.eval()
# all_references = []
# all_candidates = []
#
# with torch.no_grad():
#     for idx, d in tqdm(enumerate(val_loader), total=len(val_loader)):
#         src, attn = d
#
#         pred, pred_attn = model.predict(src[:, :6], attn[:, :6])
#         references = pred.cpu().numpy().tolist()
#         candidates = src.cpu().detach().tolist()
#
#         all_references.extend(references)
#         all_candidates.extend(candidates)
#
# bleu_score = corpus_bleu([[r] for r in all_references], [c for c in all_candidates])
# print(bleu_score)
# if os.path.exists("1.json"):
#     d = json.load(open("1.json"))
#     print(json.load(open("1.json")) == [all_references, all_candidates])
# json.dump([all_references, all_candidates], open("1.json", "w+"))
#
#
# all_references_2 = []
# all_candidates_2 = []
# model.use_kv = True
# with torch.no_grad():
#     for idx, d in tqdm(enumerate(val_loader), total=len(val_loader)):
#         src, attn = d
#
#         pred, pred_attn = model.predict_kv(src[:, :6], attn[:, :6])
#         references = pred.cpu().numpy().tolist()
#         candidates = src.cpu().detach().tolist()
#
#         all_references_2.extend(references)
#         all_candidates_2.extend(candidates)
#
# bleu_score_2 = corpus_bleu([[r] for r in all_references_2], [c for c in all_candidates_2])
# print(bleu_score_2)
# if os.path.exists("2.json"):
#     print(json.load(open("2.json")) == [all_references_2, all_candidates_2])
# json.dump([all_references_2, all_candidates_2], open("2.json", "w+"))
