import cv2
import json
import os
from tqdm import tqdm
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
import math

# 경로 설정
video_folder = "new_Kra"
json_output_folder = "new_JSON"
vis_output_folder = "visualized"
os.makedirs(json_output_folder, exist_ok=True)
os.makedirs(vis_output_folder, exist_ok=True)

# 모델 초기화
yolo_model = YOLO("yolov8x.pt", verbose=False)
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True)

def extract_pose_with_padding(frame, bbox, scale=1.3):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    x1_new = int(max(cx - bw / 2, 0))
    y1_new = int(max(cy - bh / 2, 0))
    x2_new = int(min(cx + bw / 2, w))
    y2_new = int(min(cy + bh / 2, h))
    crop = frame[y1_new:y2_new, x1_new:x2_new]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_estimator.process(crop_rgb)
    if not result.pose_landmarks:
        return None
    return [
        [float(lm.x * (x2_new - x1_new) + x1_new),
         float(lm.y * (y2_new - y1_new) + y1_new),
         float(lm.visibility)]
        for lm in result.pose_landmarks.landmark
    ]

def center_of(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

for video_file in tqdm(video_files, desc="Generating JSON and Visuals"):
    video_path = os.path.join(video_folder, video_file)
    json_output_path = os.path.join(json_output_folder, video_file.replace(".mp4", ".json"))
    vis_output_path = os.path.join(vis_output_folder, video_file)

    label = 1 if video_file.startswith("Violent_") else 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 5  # 매 5프레임마다 처리
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(vis_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    annotations = {}
    frame_idx = 0
    prev_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis_frame = frame.copy()
        frame_annots = []

        if frame_idx % interval == 0:
            results = yolo_model(frame, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()

            selected_box = None
            selected_center = None

            if len(boxes) > 0:
                if prev_center is None:
                    # 첫 프레임: 가장 큰 confidence (또는 임의 첫 bbox)
                    selected_box = boxes[0].astype(int)
                    selected_center = center_of(selected_box)
                else:
                    # 이후: 이전 중심과 가장 가까운 bbox 선택
                    min_dist = float('inf')
                    for box in boxes:
                        c = center_of(box)
                        d = distance(prev_center, c)
                        if d < min_dist:
                            min_dist = d
                            selected_box = box.astype(int)
                            selected_center = c

            if selected_box is not None:
                pose = extract_pose_with_padding(frame, selected_box)
                if pose:
                    prev_center = selected_center  # update tracking
                    x1, y1, x2, y2 = [int(v) for v in selected_box]
                    frame_annots.append({
                        "bbox": [x1, y1, x2, y2],
                        "keypoints": pose,
                        "label": int(label)
                    })
                    # 시각화
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for px, py, v in pose:
                        if v > 0.5:
                            cv2.circle(vis_frame, (int(px), int(py)), 3, (0, 0, 255), -1)
                    for i1, i2 in POSE_CONNECTIONS:
                        x1_, y1_, v1 = pose[i1]
                        x2_, y2_, v2 = pose[i2]
                        if v1 > 0.5 and v2 > 0.5:
                            cv2.line(vis_frame, (int(x1_), int(y1_)), (int(x2_), int(y2_)), (255, 0, 0), 2)

        if frame_annots:
            annotations[f"Frame_{frame_idx:07d}"] = frame_annots

        out.write(vis_frame)
        frame_idx += 1

    with open(json_output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    cap.release()
    out.release()

pose_estimator.close()
