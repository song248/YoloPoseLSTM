import cv2
import json
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# 경로 설정
video_path = 'Violent_00342.mp4'
json_path = 'Violent_00342.json'
output_dir = 'skeleton_sequences'
os.makedirs(output_dir, exist_ok=True)

# JSON 로드
with open(json_path, 'r') as f:
    annotations = json.load(f)

# 비디오 로드
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 결과 저장용
sequence_length = 30
temp_sequence = []
temp_labels = []
result_count = 0

# 프레임 단위로 반복
for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    key = f"Frame_{frame_idx:07d}"
    if key not in annotations:
        continue

    data = annotations[key]["pedestriansData"][0]
    x1, y1, x2, y2, label_str = int(data[0]), int(data[1]), int(data[2]), int(data[3]), data[4]
    label = 1 if label_str == "Violent" else 0

    # Crop & pose 추출
    person_img = frame[y1:y2, x1:x2]
    person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    result = pose.process(person_rgb)

    if result.pose_landmarks:
        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        temp_sequence.append(landmarks)
        temp_labels.append(label)

    # 일정 길이의 시퀀스를 구성
    if len(temp_sequence) == sequence_length:
        seq_array = np.array(temp_sequence)  # (30, 33, 3)
        majority_label = int(np.mean(temp_labels) >= 0.5)

        np.save(os.path.join(output_dir, f'seq_{result_count:04d}.npy'), seq_array)
        with open(os.path.join(output_dir, f'seq_{result_count:04d}_label.txt'), 'w') as f:
            f.write(str(majority_label))

        temp_sequence = []
        temp_labels = []
        result_count += 1

cap.release()
pose.close()
print("✅ 시퀀스 생성 완료!")
