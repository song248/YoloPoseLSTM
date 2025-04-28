import cv2
import json
import numpy as np
import mediapipe as mp
import os
from glob import glob

# 설정
video_path = "Violent_00628.mp4"
json_path = "Violent_00628.json"
seq_folder = "skeleton_sequences"  # 시퀀스 파일(.npy, .txt)이 저장된 폴더
output_video_path = "Violent_00628_visual.mp4"

# Mediapipe connections
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# 영상 정보 추출
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# JSON 로딩
with open(json_path, 'r') as f:
    annotations = json.load(f)

# 시퀀스 파일 정렬 로딩
sequence_files = sorted(glob(os.path.join(seq_folder, "seq_*.npy")))
label_files = [f.replace(".npy", "_label.txt") for f in sequence_files]

# 시작 프레임 추정
frame_keys = sorted(annotations.keys())
frame_indices = [int(k.split('_')[-1]) for k in frame_keys]
start_frame = min(frame_indices)

# 영상 재로딩
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_counter = start_frame

# 시퀀스별 시각화
for seq_file, label_file in zip(sequence_files, label_files):
    skeleton_data = np.load(seq_file)  # shape: (30, 33, 3)
    with open(label_file, 'r') as f:
        label = int(f.read().strip())

    # 색상 설정
    color = (0, 0, 255) if label == 1 else (0, 255, 0)

    for i in range(skeleton_data.shape[0]):
        ret, frame = cap.read()
        if not ret:
            break

        key = f"Frame_{frame_counter:07d}"
        if key not in annotations:
            frame_counter += 1
            continue

        # bbox 정보로 crop 영역 기준 좌표 복원
        data = annotations[key]["pedestriansData"][0]
        x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        crop_w, crop_h = x2 - x1, y2 - y1
        landmarks = skeleton_data[i]

        # 포인트 복원
        points = []
        for (x, y, z) in landmarks:
            px = int(x * crop_w) + x1
            py = int(y * crop_h) + y1
            points.append((px, py))
            cv2.circle(frame, (px, py), 3, color, -1)

        # 관절 연결
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], color, 1)

        out.write(frame)
        frame_counter += 1

cap.release()
out.release()
print("✅ 영상 저장 완료:", output_video_path)
