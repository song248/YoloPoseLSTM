import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 관절 그룹 설정 (Mediapipe 기준)
GROUPS = [
    [12, 14, 16], [11, 13, 15],
    [14, 16, 20], [13, 15, 19],
    [24, 26, 28], [23, 25, 27],
    [26, 28, 32], [25, 27, 31]
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_features(keypoints):
    features = []
    for group in GROUPS:
        a, b, c = keypoints[group[0]], keypoints[group[1]], keypoints[group[2]]
        if a[2] < 0.2 or b[2] < 0.2 or c[2] < 0.2:
            return None
        angle = calculate_angle(a[:2], b[:2], c[:2])
        confidence = (a[2] + b[2] + c[2]) / 3
        features.append(angle)
        features.append(confidence)
    return features  # 16차원

class PoseSequenceDataset(Dataset):
    def __init__(self, json_folder, sequence_length=6):
        self.data, self.labels = [], []
        for file in os.listdir(json_folder):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(json_folder, file), 'r') as f:
                ann = json.load(f)

            frames = sorted([int(k.replace("Frame_", "")) for k in ann.keys()])
            label = next((p[0]["label"] for k, p in ann.items() if p), 0)

            for i in range(len(frames) - sequence_length + 1):
                group = frames[i:i+sequence_length]
                if all(group[j+1] - group[j] == 5 for j in range(sequence_length - 1)):
                    sequence = []
                    valid = True
                    for fidx in group:
                        key = f"Frame_{fidx:07d}"
                        keypoints = ann[key][0]["keypoints"]
                        fvec = extract_features(keypoints)
                        if fvec is None:
                            valid = False
                            break
                        sequence.append(fvec)
                    if valid:
                        self.data.append(sequence)
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class FightLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return torch.sigmoid(out).squeeze()

def train(model, train_loader, val_loader, epochs=1000, lr=1e-4, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    early_stop_counter = 0

    train_loss_log, val_loss_log = [], []
    train_acc_log, val_acc_log = [], []
    val_precision_log, val_recall_log, val_f1_log = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out > 0.5).eq(y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()
                pred = (out > 0.5).float()
                correct += pred.eq(y).sum().item()
                all_labels += y.cpu().tolist()
                all_preds += pred.cpu().tolist()

        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        train_acc_log.append(train_acc)
        val_acc_log.append(val_acc)
        val_precision_log.append(val_precision)
        val_recall_log.append(val_recall)
        val_f1_log.append(val_f1)

        print(f"[{epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "model_best.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    # 시각화
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.plot(train_loss_log, label="Train Loss")
    plt.plot(val_loss_log, label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(train_acc_log, label="Train Acc")
    plt.plot(val_acc_log, label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(val_precision_log, label="Val Precision")
    plt.title("Precision")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(val_recall_log, label="Val Recall")
    plt.title("Recall")
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(val_f1_log, label="Val F1")
    plt.title("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

if __name__ == "__main__":
    json_folder = "new_JSON"

    dataset = PoseSequenceDataset(json_folder)
    print(f"총 시퀀스 수: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = FightLSTM()
    train(model, train_loader, val_loader)
