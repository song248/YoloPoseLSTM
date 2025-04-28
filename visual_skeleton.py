import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# 설정
video_path = "test/Videos/Violent_00671.mp4"
annotation_json_path = "test/JSON/Violent_00671.json"
output_video_path = "Violent_00671_pose_visualized.mp4"

# Mediapipe pose connections (관절 연결선)
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# JSON 로딩
with open(annotation_json_path, 'r') as f:
    annotations = json.load(f)

# 비디오 읽기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0

pbar = tqdm(total=total_frames, desc="Visualizing")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = f"Frame_{frame_idx:07d}"
    if key in annotations:
        frame_annos = annotations[key]
        for person in frame_annos:
            bbox = person["bbox"]
            keypoints = np.array(person["keypoints"])
            label = person["label"]
            color = (0, 255, 0) if label == 0 else (0, 0, 255)  # Normal=Green, Violent=Red

            # 관절 점 그리기
            for (x, y, z) in keypoints:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

            # 관절 연결선 그리기
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, color, 1)

            # bbox도 그리기 (선택)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

            # label 텍스트 쓰기
            label_text = "Normal" if label == 0 else "Violent"
            cv2.putText(frame, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out.write(frame)
    frame_idx += 1
    pbar.update(1)

cap.release()
out.release()
pbar.close()

print(f"✅ 시각화된 영상 저장 완료: {output_video_path}")
