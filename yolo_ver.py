from ultralytics import YOLO
import cv2

# 모델 로드 (YOLOv8s - 경량 버전)
model = YOLO("yolov8l.pt")  # 필요에 따라 yolov8m.pt, yolov8l.pt 등 선택 가능

# 영상 경로
video_path = "normal/Normal_01336.mp4"
output_path = "output_with_boxes.mp4"

# 영상 읽기
cap = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# 결과 저장을 위한 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 추론
    results = model(frame)

    # 시각화 및 필터링 (사람 클래스: class==0 in COCO)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 0 = person in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
