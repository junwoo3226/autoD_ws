import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from ultralytics import YOLO
from playsound import playsound
import threading

class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # YOLO 모델 로드
        self.model = YOLO('/home/jw/odegi_ws/src/odegi_fire/odegi_fire/best.pt')  # 학습된 모델 파일 경로

        # OpenCV 웹캠 설정
        self.cap = cv2.VideoCapture(0)  # 0은 기본 웹캠
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            sys.exit()

        # PyQt 창 설정
        self.setWindowTitle("YOLOv8 Real-Time Detection with Alarm and AMR Button")
        self.setGeometry(100, 100, 800, 600)

        # QLabel을 이용해 비디오 프레임 표시
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # AMR 출동 버튼 추가
        self.amr_button = QPushButton("AMR 출동", self)
        self.amr_button.setFixedHeight(40)
        self.amr_button.setStyleSheet("font-size: 16px; background-color: #4CAF50; color: white;")
        self.amr_button.clicked.connect(self.amr_button_clicked)

        # 레이아웃 설정
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.video_label)
        layout.addWidget(self.amr_button)
        self.setCentralWidget(central_widget)

        # QTimer로 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 간격으로 프레임 업데이트

        # 알람 및 반짝임 상태 변수
        self.alarm_triggered = False
        self.flash_state = False  # 반짝임 상태

        # 구역 설정 (12개 구역)
        self.GRID_WIDTH = 160
        self.GRID_HEIGHT = 160
        self.zones = [
            (x * self.GRID_WIDTH, y * self.GRID_HEIGHT, (x + 1) * self.GRID_WIDTH, (y + 1) * self.GRID_HEIGHT)
            for y in range(3) for x in range(4)
        ]

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("웹캠 프레임을 읽을 수 없습니다.")
            return

        # YOLO 추론
        results = self.model.predict(source=frame, conf=0.5, save=False, verbose=False)

        # 감지 결과 시각화
        annotated_frame = results[0].plot()

        # 구역 표시
        for zone_num, (sx1, sy1, sx2, sy2) in enumerate(self.zones, start=1):
            cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), (255, 0, 0), 1)

        # 탐지된 객체에 대한 정보 표시 및 구역 판별
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls)
                class_name = self.model.names[cls]

                # 바운딩 박스와 라벨 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 탐지된 객체의 중심 좌표 계산
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 탐지된 객체가 어느 구역에 있는지 판별
                for zone_num, (sx1, sy1, sx2, sy2) in enumerate(self.zones, start=1):
                    if sx1 <= cx <= sx2 and sy1 <= cy <= sy2 and conf > 0.75:
                        print(f"{zone_num}번 구역에 {class_name} 감지됨! (신뢰도: {conf:.2f})")
                        break

        # OpenCV BGR 이미지를 RGB로 변환
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # PyQt용 QImage로 변환
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # QLabel에 QPixmap으로 표시
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

        # 감지된 객체에서 특정 조건 확인
        for detection in results[0].boxes.data:
            class_id = int(detection[5].item())  # 클래스 ID
            confidence = detection[4].item()  # 신뢰도

            # 조건: 클래스 ID가 0(fire)이고 신뢰도가 0.75 이상인 경우 알람 및 반짝임
            if class_id == 0 and confidence > 0.75:  # 'fire' 클래스
                if not self.alarm_triggered:
                    self.trigger_alarm_and_flash()
                break

    def trigger_alarm_and_flash(self):
        self.alarm_triggered = True
        threading.Thread(target=self.play_alarm).start()
        self.flash_screen()

    def play_alarm(self):
        try:
            playsound('/home/jw/odegi_ws/src/odegi_fire/odegi_fire/notification.wav')  # 알람 사운드 파일 경로
        except Exception as e:
            print(f"알람 재생 중 오류 발생: {e}")
        self.alarm_triggered = False  # 알람 재생 후 리셋

    def flash_screen(self):
        if self.flash_state:  # 반짝임 동작 중복 방지
            return

        def toggle_flash():
            self.flash_state = True
            for _ in range(10):  # 반짝임 반복 횟수
                color = "red" if self.flash_state else "white"
                self.video_label.setStyleSheet(f"background-color: {color};")
                self.flash_state = not self.flash_state
                QTimer.singleShot(100, lambda: None)  # 100ms 간격으로 반짝임
            self.flash_state = False

        threading.Thread(target=toggle_flash).start()

    def amr_button_clicked(self):
        # AMR 버튼 클릭 시 실행되는 동작 (추후 기능 추가 가능)
        print("AMR 출동 버튼이 클릭되었습니다!")

    def closeEvent(self, event):
        # 창 닫힐 때 자원 해제
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


def main():
    """PyQt 앱 실행을 위한 메인 함수"""
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()