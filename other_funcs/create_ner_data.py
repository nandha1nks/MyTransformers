import json
from ast import literal_eval
import pandas as pd

df = pd.read_csv("../data/ner.csv")

data = []
for idx, row in df.iterrows():
    if idx >= 64:
        break
    tags = literal_eval(row['Tag'])
    sentence = row['Sentence']

    words = sentence.split()
    di = {}
    for word, tag in zip(words, tags):
        if tag == 'O':
            continue
        if tag[0] == 'B':
            di[tag[2:]] = di.get(tag[2:], []) + [word]
        else:
            di[tag[2:]][-1] += f" {word}"
    data.append({
        "sentence": sentence,
        "ner": di})

json.dump(data, open("../data/ner.json", "w+"))
