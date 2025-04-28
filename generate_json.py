import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import mediapipe as mp

# 설정
video_path = "Violent_00628.mp4"  # 또는 Violent_00008.mp4
source_json_path = "Violent_00628.json"  # 정답 bbox+label JSON
annotation_output_path = "Violent_00628_pose_annotation.json"

# ✅ YOLO11 m 모델 로드
yolo_model = YOLO("yolo11m.pt")  # COCO pre-trained YOLO11m 모델

# Mediapipe pose 설정
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True)

# 정답 JSON 로딩
with open(source_json_path, 'r') as f:
    gt_annotations = json.load(f)

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
    padding = 10  # bbox 주변에 패딩 추가 (관절 잘림 방지)
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
    return ratio >= threshold  # 80% 이상 관절이 bbox 안에 있어야 통과

# 메인
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps // 5)  # 초당 5프레임
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_annotations = {}
frame_idx = 0

pbar = tqdm(total=frame_count, desc="Processing frames")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % interval != 0:
        frame_idx += 1
        pbar.update(1)
        continue

    results = yolo_model(frame)[0]  # ✅ YOLO11m inference

    frame_annotations = []
    key = f"Frame_{frame_idx:07d}"

    gt_bboxes = []
    gt_labels = []
    if key in gt_annotations:
        for ann in gt_annotations[key]["pedestriansData"]:
            gt_bboxes.append(list(map(int, ann[:4])))
            gt_labels.append(1 if ann[4] == "Violent" else 0)

    for det_idx, det in enumerate(results.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = det.astype(int)
        det_box = [x1, y1, x2, y2]

        matched_label = 0  # 기본은 Normal
        matched_gt_bbox = None
        if gt_bboxes:
            for idx, gt_box in enumerate(gt_bboxes):
                iou = compute_iou(det_box, gt_box)
                if iou > 0.4:  # ✅ IoU 기준 완화
                    matched_label = gt_labels[idx]
                    matched_gt_bbox = gt_box
                    break

        if matched_gt_bbox is None:
            continue  # 정답 매칭 실패 시 패스

        pose = extract_pose(frame, det_box)
        if pose and check_keypoints_in_bbox(pose, matched_gt_bbox):
            frame_annotations.append({
                "person_id": f"person_{int(x1)}_{int(y1)}",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "keypoints": pose,  # 이미 float 변환 완료
                "label": int(matched_label)
            })

    if frame_annotations:
        all_annotations[key] = frame_annotations

    frame_idx += 1
    pbar.update(1)

cap.release()
pose_estimator.close()
pbar.close()

# 최종 JSON 저장
with open(annotation_output_path, 'w') as f:
    json.dump(all_annotations, f, indent=2)

print(f"✅ Pose annotation JSON 저장 완료: {annotation_output_path}")
