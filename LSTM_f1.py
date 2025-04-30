import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

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
        a = keypoints[group[0]]
        b = keypoints[group[1]]
        c = keypoints[group[2]]

        if a[2] < 0.2 or b[2] < 0.2 or c[2] < 0.2:
            return None

        angle = calculate_angle(a[:2], b[:2], c[:2])
        confidence = (a[2] + b[2] + c[2]) / 3

        features.append(angle)
        features.append(confidence)
    return features

class PoseSequenceDataset(Dataset):
    def __init__(self, json_folder, sequence_length=15):
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
                    if features is not None:
                        features_seq.append(features)

            label = 0
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

class FightLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2, dropout=0.5):
        super(FightLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()

def train(model, train_loader, val_loader, epochs=1000, lr=1e-4, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_precisions, val_recalls, val_f1s = [], [], []

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
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)

        epoch_pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}",
            "Train Acc": f"{train_accuracy:.4f}",
            "Val Acc": f"{val_accuracy:.4f}",
            "F1": f"{f1:.4f}"
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'model/best_model.pth')
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(val_precisions, label='Precision')
    plt.plot(val_recalls, label='Recall')
    plt.plot(val_f1s, label='F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Precision/Recall/F1')

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
    train(model, train_loader, val_loader, epochs=1000, lr=1e-4, patience=50)
