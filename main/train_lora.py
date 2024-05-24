import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline

from model.lora import make_model_for_lora, get_trainable_parameters
from other_funcs.utils import convert_transformers_conv1d_to_linear
from nltk.translate.bleu_score import corpus_bleu

from utils.datasets import NERDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def change_and_load_model(base_model, change_layer, weight_path=None, *args, **kwargs):
    change_layer(base_model, *args, **kwargs)
    if weight_path:
        base_model.load_state_dict(torch.load(weight_path))


def generate(model_, src, max_tokens, eos_token):
    src = src.copy()

    input_ids = src["input_ids"]
    attn_mask = src["attention_mask"]
    position_ids = src["position_ids"]

    batch_size = input_ids.size(0)
    inp_token = input_ids.size(1)

    last_tokens = torch.zeros((batch_size,), dtype=torch.long)
    c = 0
    while not last_tokens.all() and c < max_tokens:
        with torch.no_grad():
            d = model_(input_ids=input_ids, attention_mask=attn_mask, position_ids=position_ids)
        last_token = d.logits[:, -1, :].argmax(dim=-1).to(torch.long)
        last_token[last_tokens == 1] = eos_token
        last_token = last_token.reshape(-1, 1)

        input_ids = torch.concat([input_ids, last_token], dim=-1)
        last_tokens = last_tokens.masked_fill(last_token.reshape((-1,)) == eos_token, 1)
        attn_mask = torch.concat([attn_mask, (last_tokens.reshape(-1, 1) + 1) % 2], dim=-1)
        position_ids = torch.concat([position_ids, position_ids[:, -1:] + attn_mask[:, -1:]], dim=-1)

        c += 1

    answers = []
    for i in range(batch_size):
        answers.append(input_ids[i, inp_token:][attn_mask[i, inp_token:] == 1].cpu().detach().tolist())

    return answers


def evaluate(model_, data_loader):
    all_references = []
    all_candidates = []
    for data in data_loader:
        src, tgt = data
        max_tokens = max(len(t) for t in tgt)
        pred = generate(model_, src, max_tokens * 3, tokenizer.eos_token_id)
        all_references.extend([tokenizer.decode(x) for x in pred])
        all_candidates.extend([tokenizer.decode(x) for x in tgt['input_ids']])
    print("\n".join(random.sample(all_references, 5)))
    bleu_score = corpus_bleu([[r] for r in all_references], [c for c in all_candidates])
    return bleu_score


pipe = pipeline("text-generation", model="openai-community/gpt2")
model = pipe.model
tokenizer = pipe.tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

change_and_load_model(model, convert_transformers_conv1d_to_linear)


def val_collate(batch):
    tokenizer.padding_side = "left"
    sents = [x[0] for x in batch]
    tgts = [x[1] + " " + tokenizer.eos_token for x in batch]

    src = tokenizer(sents, return_tensors="pt", padding=True).to(device)
    tgt = tokenizer(tgts, padding=False)
    pos_ids = torch.zeros_like(src['attention_mask']).to(device)
    for i in range(src['attention_mask'].size(0)):
        f = src['attention_mask'][i, :].sum()
        pos_ids[i][src['attention_mask'].size(1)-f:] = torch.arange(0, f)
    src['position_ids'] = pos_ids

    return src, tgt


def train_collate(batch):
    tokenizer.padding_side = "right"
    sents = [x[0] + " " + x[1] + " " + tokenizer.eos_token
             for x in batch]
    return tokenizer(sents, return_tensors="pt", padding=True).to(device)


dataset = NERDataset("../data/ner.json")
loader = DataLoader(dataset, collate_fn=val_collate, batch_size=4)
train_loader = DataLoader(dataset, collate_fn=train_collate, batch_size=4)

print(f"Trainable parameters before LORA config: {get_trainable_parameters(model)}")
make_model_for_lora(model, 32, 32)
print(f"Trainable parameters after LORA config: {get_trainable_parameters(model)}")

# model.load_state_dict(torch.load("lora.pth"))
# print(evaluate(model, loader))

model = model.to(device)

optimizer_lr = 2e-5
betas = (0.9, 0.98)
eps = 1e-9
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=optimizer_lr,
    betas=betas,
    eps=eps
)


epoch = 50
for e in tqdm(range(epoch)):
    model.train()
    train_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        input_ids = data["input_ids"]
        attn_mask = data["attention_mask"]

        d = model(input_ids=input_ids[:, :-1])
        flattened_mask = attn_mask[:, :-1].reshape(-1)
        loss = criterion(d.logits.view(-1, d.logits.size(-1))[flattened_mask == 1],
                         input_ids[:, 1:].reshape(-1)[flattened_mask == 1])
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    average_train_loss = train_loss / len(train_loader)

    torch.save(model.state_dict(), "lora.pth")
    print(f"Train Loss: {round(average_train_loss, 5)}\n Bleu: {round(evaluate(model, loader), 5)}")
