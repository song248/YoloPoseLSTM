import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 관절 그룹 설정 (Mediapipe 33개 기준)
GROUPS = [
    [12, 14, 16],
    [11, 13, 15],
    [14, 16, 20],
    [13, 15, 19],
    [24, 26, 28],
    [23, 25, 27],
    [26, 28, 32],
    [25, 27, 31]
]

# 각도 계산 함수
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Feature 추출 함수
def extract_features(keypoints):
    features = []
    for group in GROUPS:
        a = keypoints[group[0]]
        b = keypoints[group[1]]
        c = keypoints[group[2]]

        angle = calculate_angle(a[:2], b[:2], c[:2])
        confidence = (a[2] + b[2] + c[2]) / 3

        features.append(angle)
        features.append(confidence)
    return features

# 커스텀 Dataset
class PoseSequenceDataset(Dataset):
    def __init__(self, json_folder, sequence_length=30):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length

        for json_file in os.listdir(json_folder):
            if not json_file.endswith('.json'):
                continue
            path = os.path.join(json_folder, json_file)
            with open(path, 'r') as f:
                annotation = json.load(f)

            frames = sorted(annotation.keys())
            features_seq = []

            for frame in frames:
                persons = annotation[frame]
                if isinstance(persons, list) and persons:
                    keypoints = persons[0]['keypoints']
                    features = extract_features(keypoints)
                    features_seq.append(features)

            label = 0  # 기본값
            for frame in frames:
                persons = annotation[frame]
                if isinstance(persons, list) and persons:
                    label = persons[0]['label']
                    break

            for i in range(0, len(features_seq) - sequence_length + 1, sequence_length):
                self.data.append(features_seq[i:i+sequence_length])
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# LSTM 모델
class FightLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2):
        super(FightLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()

# 학습 함수
def train(model, train_loader, val_loader, epochs=1000, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    epoch_pbar = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        epoch_pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}",
            "Train Acc": f"{train_accuracy:.4f}",
            "Val Acc": f"{val_accuracy:.4f}"
        })

    # 학습 완료 후 그래프 저장
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    json_folder = "JSON"
    dataset = PoseSequenceDataset(json_folder)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = FightLSTM()
    train(model, train_loader, val_loader)
