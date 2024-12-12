import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-3, weight_decay=1e-5, num_epochs=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # 可选学习率调度器
        self.scheduler = None

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm进度条
        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.num_epochs}"):
            claim = batch['claim'].to(self.device)  # (batch_size, claim_len, input_dim)
            relevant_articles = batch['relevant_articles'].to(self.device) # (batch_size, num_articles, article_len, input_dim)
            labels = batch['labels'].to(self.device)  # (batch_size,)

            self.optimizer.zero_grad()
            predictions, loss = self.model(claim, relevant_articles, labels=labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * claim.size(0)
            # 计算准确率
            preds = predictions.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc=f"Evaluating Epoch {epoch+1}/{self.num_epochs}"):
            claim = batch['claim'].to(self.device)
            relevant_articles = batch['relevant_articles'].to(self.device)
            labels = batch['labels'].to(self.device)

            predictions, loss = self.model(claim, relevant_articles, labels=labels)
            total_loss += loss.item() * claim.size(0)

            preds = predictions.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate(epoch)

            # 如果有学习率调度器，可以在这里调用 step()
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                print("Best model saved.")

        print("Training finished.")
