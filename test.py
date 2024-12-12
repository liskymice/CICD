import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

tsv_file = "./datasets/Politifact/politifact.tsv"
input_dim = 768      # 使用BERT输出768维隐藏层
hidden_dim = 120
vocab_size = 100
num_classes = 3
top_k = 5
max_len = 20
batch_size = 128
num_epochs = 10
lr = 2e-3

max_seq_length = 50   # 对claim和article都截断或padding到50个token

from transformers import AutoTokenizer, AutoModel

# 初始化BERT tokenizer和model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.eval()  # 推理模式


def encode_text(text):
    """
    使用BERT将文本encode为 (seq_len, 768) 的嵌入。
    seq_len固定为max_seq_length。
    """
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length',
                                  max_length=max_seq_length)
        outputs = bert_model(**encoded_input)
        # outputs.last_hidden_state: (1, seq_len, 768)
        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)
    return embeddings.cpu().float()  # 返回Tensor类型，方便后续stack


class PolitifactDataset(Dataset):
    def __init__(self, tsv_file):
        self.label_map = {
            "true": 0,
            "mixed": 1,
            "false": 2
        }
        self.data = self._read_tsv_and_group(tsv_file)

    def _normalize_label(self, correctness):
        correctness = correctness.strip().lower()
        if correctness == "true":
            return "true"
        elif correctness in ["mostly-true", "half-true", "mostly-false"]:
            return "mixed"
        elif correctness in ["false", "pants-fire"]:
            return "false"
        else:
            return "false"

    def _read_tsv_and_group(self, tsv_file):
        data_dict = {}
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                correctness = self._normalize_label(row[0])
                claim = row[2].strip()
                article = row[4].strip()
                label_id = self.label_map[correctness]

                if claim not in data_dict:
                    data_dict[claim] = {
                        "label": label_id,
                        "articles": [article]
                    }
                else:
                    data_dict[claim]["articles"].append(article)

        all_data = []
        for c, info in data_dict.items():
            all_data.append({
                "claim": c,
                "label": info["label"],
                "articles": info["articles"]
            })
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item["claim"]
        label = item["label"]
        articles = item["articles"]

        # 对claim编码 (seq_len, 768)
        claim_tensor = encode_text(claim)  # now it's a Tensor

        # 对articles编码，固定取前5篇，没有则padding空字符串
        num_articles = 5
        selected_articles = articles[:num_articles]
        if len(selected_articles) < num_articles:
            selected_articles += [""] * (num_articles - len(selected_articles))

        article_tensors = []
        for art in selected_articles:
            art_tensor = encode_text(art)  # (seq_len, 768)
            article_tensors.append(art_tensor)
        # 堆叠 (num_articles, seq_len, 768)
        article_tensors = torch.stack(article_tensors, dim=0)

        return claim_tensor, article_tensors, label


def collate_fn(batch):
    # batch是list，元素是 (claim_tensor, article_tensor, label)
    claims = [b[0] for b in batch]  # list of (seq_len, 768) tensors
    articles = [b[1] for b in batch]  # list of (num_articles, seq_len, 768)
    labels = [b[2] for b in batch]

    claims = torch.stack(claims, dim=0)  # (batch_size, seq_len, 768)
    articles = torch.stack(articles, dim=0)  # (batch_size, num_articles, seq_len, 768)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "claim": claims,
        "relevant_articles": articles,
        "labels": labels
    }

dataset = PolitifactDataset(tsv_file)
total_len = len(dataset)
train_len = int(total_len * 0.8)
val_len = total_len - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

from Models.CICD import CICD
from Trainer import Trainer

model = CICD(input_dim=input_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, top_k=top_k, num_classes=num_classes, max_len=max_len)

trainer = Trainer(model, train_loader, val_loader, device='cuda', lr=lr, num_epochs=num_epochs)

trainer.fit()
