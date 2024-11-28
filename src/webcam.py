import cv2
from ultralytics import YOLO
# 1. 학습된 YOLOv8 모델 로드 (사용자 모델 경로)
model_path = "best.pt"  # 학습한 모델 파일 경로
model = YOLO(model_path)
# 2. 웹캠 캡처 객체 생성 (USB 웹캠 사용 시 인덱스 변경 가능)
cap = cv2.VideoCapture(0)  # 기본 웹캠(0), USB 웹캠(2) 등
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()
# 3. 실시간 객체 탐지 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    # 4. YOLOv8 객체 탐지 수행
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)
    # 5. 탐지된 객체에 대한 바운딩 박스 및 라벨 표시
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            conf = box.conf[0]  # 신뢰도
            cls = int(box.cls)  # 클래스 인덱스
            class_name = model.names[cls]  # 클래스 이름
            # 바운딩 박스와 라벨 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 6. 프레임 출력
    cv2.imshow("YOLOv8 Webcam Detection", frame)
    # 7. 'q' 키 입력 시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 8. 리소스 해제
cap.release()
cv2.destroyAllWindows()