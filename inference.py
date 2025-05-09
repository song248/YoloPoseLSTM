import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from train import FightLSTM, extract_features, GROUPS  # 기존 train.py와 연동

# 시퀀스 생성 (5프레임 간격, 6개 연속)
def generate_sequences(annotation, sequence_length=6):
    frames = sorted([int(f.replace("Frame_", "")) for f in annotation])
    sequences = []

    for i in range(len(frames) - sequence_length + 1):
        group = frames[i:i+sequence_length]
        if all(group[j+1] - group[j] == 5 for j in range(sequence_length - 1)):
            sequence = []
            valid = True
            for fidx in group:
                key = f"Frame_{fidx:07d}"
                keypoints = annotation[key][0]['keypoints']
                fvec = extract_features(keypoints)
                if fvec is None:
                    valid = False
                    break
                sequence.append(fvec)
            if valid:
                sequences.append(sequence)
    return sequences  # list of [6 x 16]

def classify_json(model, json_path, device):
    with open(json_path, 'r') as f:
        annotation = json.load(f)

    sequences = generate_sequences(annotation)
    if not sequences:
        return "정상"  # 데이터가 없으면 기본값

    with torch.no_grad():
        for seq in sequences:
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(x)
            if out.item() > 0.5:
                return "폭력"

    return "정상"

def main():
    test_folder = "test"
    model_path = "model_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FightLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("📁 테스트셋 결과:")
    for fname in os.listdir(test_folder):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(test_folder, fname)
        result = classify_json(model, fpath, device)
        print(f"{fname}: {result}")

if __name__ == "__main__":
    main()
