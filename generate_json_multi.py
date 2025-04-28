import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import mediapipe as mp

# 설정
# base_path = "test"
base_path = "Kranok-NV"
video_folder = os.path.join(base_path, "Videos")
annotation_folder = os.path.join(base_path, "Annotations")
output_folder = "JSON"

os.makedirs(output_folder, exist_ok=True)

# YOLO11m 모델 로드 (verbose=False로 설정하여 출력 억제)
yolo_model = YOLO("yolo11m.pt", verbose=False)

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True)

# IoU 계산 함수
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

# 관절 추출 함수
def extract_pose(frame, bbox):
    x1, y1, x2, y2 = bbox
    padding = 10
    h, w = frame.shape[:2]
    x1_p = max(x1 - padding, 0)
    y1_p = max(y1 - padding, 0)
    x2_p = min(x2 + padding, w)
    y2_p = min(y2 + padding, h)
    crop = frame[y1_p:y2_p, x1_p:x2_p]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_estimator.process(crop_rgb)
    if not result.pose_landmarks:
        return None
    landmarks = []
    for lm in result.pose_landmarks.landmark:
        landmarks.append([
            float(lm.x * (x2_p - x1_p) + x1_p),
            float(lm.y * (y2_p - y1_p) + y1_p),
            float(lm.z)
        ])
    return landmarks

# 관절이 bbox 안에 포함되는지 검증
def check_keypoints_in_bbox(keypoints, bbox, threshold=0.8):
    x1, y1, x2, y2 = bbox
    inside_count = 0
    for (x, y, z) in keypoints:
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside_count += 1
    ratio = inside_count / len(keypoints)
    return ratio >= threshold

# 비디오 파일 리스트 가져오기
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

video_pbar = tqdm(video_files, desc="Processing videos", unit="video")

for idx, video_file in enumerate(video_pbar, 1):
    video_pbar.set_postfix({"Progress": f"{idx}/{len(video_files)} ({(idx/len(video_files))*100:.2f}%)"})

    video_path = os.path.join(video_folder, video_file)
    json_name = os.path.splitext(video_file)[0] + ".json"
    source_json_path = os.path.join(annotation_folder, json_name)
    output_json_path = os.path.join(output_folder, json_name)

    if not os.path.exists(source_json_path):
        print(f"❌ Annotation not found for {video_file}, skipping.")
        continue

    with open(source_json_path, 'r') as f:
        gt_annotations = json.load(f)

    violence_label = 1 if video_file.startswith("Violent_") else 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps // 5)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_annotations = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval != 0:
            frame_idx += 1
            continue

        results = yolo_model(frame, verbose=False)[0]
        frame_annotations = []
        key = f"Frame_{frame_idx:07d}"

        gt_bboxes = []
        if key in gt_annotations:
            for ann in gt_annotations[key]["pedestriansData"]:
                gt_bboxes.append(list(map(int, ann[:4])))

        for det_idx, det in enumerate(results.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = det.astype(int)
            det_box = [x1, y1, x2, y2]

            matched_gt_bbox = None
            if gt_bboxes:
                for gt_box in gt_bboxes:
                    iou = compute_iou(det_box, gt_box)
                    if iou > 0.4:
                        matched_gt_bbox = gt_box
                        break

            if matched_gt_bbox is None:
                continue

            pose = extract_pose(frame, det_box)
            if pose and check_keypoints_in_bbox(pose, matched_gt_bbox):
                frame_annotations.append({
                    "person_id": f"person_{int(x1)}_{int(y1)}",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "keypoints": pose,
                    "label": violence_label
                })

        if frame_annotations:
            all_annotations[key] = frame_annotations

        frame_idx += 1

    cap.release()

    # JSON 저장
    with open(output_json_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)

pose_estimator.close()
print("🎯 All videos processed!")
