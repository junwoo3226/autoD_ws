import sys
import cv2
import os
import numpy as np
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QTextEdit, QLineEdit, QDialog, QFormLayout, QMessageBox
from PyQt5 import uic
from ultralytics import YOLO
from playsound import playsound
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import GoalStatus
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.qos import QoSProfile

from std_msgs.msg import String

from nav2_msgs.action import FollowWaypoints, NavigateToPose
import sqlite3
from datetime import datetime
import math
from geometry_msgs.msg import PoseStamped, Quaternion

# 미리 지정된 위치 정보
locations = {   # [x, y, yaw]
    "start": [0.0, 0.0],
    "wp1": [0.2377847135066986, -0.6849836707115173, -1.57],
    "wp2": [0.2377847135066986, -0.6849836707115173, -3.14],
    "section1": [-1.53310227394104, -0.178277388215065, 3.14],
    "section2": [-0.7571648955345154, -0.12823615968227386, 0.0],
    "section3": [-1.5643913745880127, -0.547355055809021, -1.57],
    "section4": [-0.7571648955345154, -0.547355055809021, 3.14]
}
# 한국어 출력을 위한 번역
dictionary = {
    "start": "대기 장소",
    "wp1": "웨이포인트 1",
    "wp2": "웨이포인트 2",
    "section1": "1번 구역",
    "section2": "2번 구역",
    "section3": "3번 구역",
    "section4": "4번 구역"
}

class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Login")
        self.setGeometry(400, 300, 300, 150)

        # ID, Password 입력 폼
        self.id_input = QLineEdit(self)
        self.id_input.setPlaceholderText("ID")
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Password")

        # 로그인 버튼
        self.login_button = QPushButton("Login", self)
        self.login_button.clicked.connect(self.check_credentials)

        # 레이아웃 설정
        layout = QFormLayout(self)
        layout.addRow("ID:", self.id_input)
        layout.addRow("Password:", self.password_input)
        layout.addWidget(self.login_button)

    def check_credentials(self):
        # ID와 비밀번호 확인
        user_id = self.id_input.text()
        password = self.password_input.text()

        if user_id == "admin" and password == "123456":
        #if user_id == "" and password == "":
            self.accept()  # 로그인 성공
        else:
            # 로그인 실패 메시지
            QMessageBox.warning(self, "Login Failed", "Invalid ID or Password")

class YOLOApp(QMainWindow):
    update_signal = pyqtSignal(str)  # GUI 업데이트 신호

    def __init__(self):
        super().__init__()
        self.path = '/home/jw/odegi_ws/src/odegi_fire/odegi_fire/'

        # ROS2 초기화 및 GuiNode 생성
        rclpy.init()
        self.node = GuiNode(self)

        self.update_signal.connect(self.update_display)

        # ROS2 노드를 별도의 스레드에서 실행
        self.thread_spin = threading.Thread(target=rclpy.spin, args=(self.node,))
        self.thread_spin.start()
        
        # YOLO 모델 로드
        self.model = YOLO(self.path + 'best.pt')  # 학습된 모델 파일 경로

        # OpenCV 웹캠 설정
        self.cap = cv2.VideoCapture(3)  # 0은 기본 웹캠
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            self.update_signal.emit("웹캠을 열 수 없습니다.")
            self.close()
            sys.exit()

        # SQLite 데이터베이스 초기화
        self.db_path = "cv.db"
        self.initialize_database()

        # PyQt 창 설정
        self.setWindowTitle("오대기 관제 시스템 모니터")
        self.setGeometry(100, 100, 1000, 800)

        # QLabel을 이용해 비디오 프레임 표시
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # 텍스트 박스
        self.message_display = QTextEdit(self)
        self.message_display.setReadOnly(True)

        self._setup_ui()

        # QTimer로 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 간격으로 프레임 업데이트

        # 알람 및 반짝임 상태 변수
        self.alarm_triggered = False
        self.flash_state = False  # 반짝임 상태

        # 구역 설정 (4개 구역)
        self.GRID_WIDTH = 320
        self.GRID_HEIGHT = 240
        self.zones = [
            (x * self.GRID_WIDTH, y * self.GRID_HEIGHT, (x + 1) * self.GRID_WIDTH, (y + 1) * self.GRID_HEIGHT)
            for y in range(2) for x in range(2)
        ]
        # 구역별 감지된 프레임 카운트용 리스트
        self.zone_detection_count = {i: 0 for i in range(1, len(self.zones)+1)}
        self.DETECTION_THRESHOLD = 20   # 연속 감지 프레임 임계값

    def _setup_ui(self):
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        button_layout = self._create_buttons()
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        top_layout.addLayout(video_layout)
        top_layout.addLayout(button_layout)
        layout.addLayout(top_layout)
        layout.addWidget(self.message_display)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _create_buttons(self):
        l_button_layout = QVBoxLayout()
        r_button_layout = QVBoxLayout()
        self._create_button("1", l_button_layout, locations["section1"])
        self._create_button("3", l_button_layout, locations["section3"])
        self._create_button("return", l_button_layout, locations["start"])
        self._create_button("2", r_button_layout, locations["section2"])
        self._create_button("4", r_button_layout, locations["section4"])
        self._create_button("stop", r_button_layout, None, self.on_stop_button_click)
        main_layout = QHBoxLayout()
        main_layout.addLayout(l_button_layout)
        main_layout.addLayout(r_button_layout)
        return main_layout

    def _create_button(self, text, layout, location=None, callback=None):
        button = QPushButton(text, self)
        button.setFixedSize(150, 150)
        button.setStyleSheet("font-size: 20px; background-color: #4CAF50; color: white;")
        if callback:
            button.clicked.connect(callback)
        elif location:
            button.clicked.connect(lambda: self.on_button_click(location))
        layout.addWidget(button)

    def keyPressEvent(self, event):
        """키보드 이벤트 처리"""
        if event.key() == Qt.Key_C:
            self.capture_frame()

    def capture_frame(self):
        """현재 프레임을 캡처하여 저장"""
        ret, frame = self.cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.png"
            cv2.imwrite(filename, frame)
            self.update_signal.emit(f"캡처 완료: {filename}")  

    def initialize_database(self):
        """SQLite 데이터베이스 초기화"""
        conn = sqlite3.connect('cv.db')
        cursor = conn.cursor()
        # 화재 감지 데이터 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone INTEGER NOT NULL,
                detected_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal INTEGER NOT NULL,
                detected_at TEXT NOT NULL
            )
        """)
        # AMR 출동 데이터 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS amr_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                executed_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def save_detection_to_db(self, zone):
        """감지된 구역과 시간을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO detections (zone, detected_at) VALUES (?, ?)", (zone, detected_at))
        conn.commit()
        conn.close()

    def save_goal_reach_to_db(self, goal):
        """AMR이 목표 지점에 도착한 시간과 지점명을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        reached_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO goal_reaches (goal, reached_at) VALUES (?, ?)", (goal, reached_at))
        conn.commit()
        conn.close()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("웹캠 프레임을 읽을 수 없습니다.")
            self.update_signal.emit("웹캠 프레임을 읽을 수 없습니다.")
            return

        # YOLO 추론
        results = self.model.predict(source=frame, conf=0.7, save=False, verbose=False)
        detected_zones = set()

        # 감지 결과 시각화
        annotated_frame = results[0].plot()

        # 구역 표시
        for zone_num, (sx1, sy1, sx2, sy2) in enumerate(self.zones, start=1):
            cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), (80, 80, 80), 1)

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
                    if sx1 <= cx <= sx2 and sy1 <= cy <= sy2 and conf > 0.5:
                        detected_zones.add(zone_num)
                        self.zone_detection_count[zone_num] += 1
                        break

        # 프레임에서 감지되지 않은 구역은 카운트 초기화
        for zone_num in self.zone_detection_count:
                if zone_num not in detected_zones:
                    self.zone_detection_count[zone_num] = 0

        # 연속 감지된 구역이 임계값을 초과할 경우 메시지 출력 및 알람과 화면 번쩍임
        for zone_num, count in self.zone_detection_count.items():
            if count >= self.DETECTION_THRESHOLD:
                print(f"{zone_num}번 구역에서 연속으로 {self.DETECTION_THRESHOLD} 프레임 이상 감지됨!")
                self.update_signal.emit(f"{zone_num}번 구역에서 화재가 감지되고 있습니다!")
                if not self.alarm_triggered:
                    self.trigger_alarm_and_flash()
                    print("알람 및 반짝임 실행")
                self.zone_detection_count[zone_num] = 0

        # OpenCV BGR 이미지를 RGB로 변환
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # PyQt용 QImage로 변환
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # QLabel에 QPixmap으로 표시
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def trigger_alarm_and_flash(self):
        self.alarm_triggered = True
        threading.Thread(target=self.play_alarm).start()
        self.flash_screen()

    def play_alarm(self):
        try:
            playsound(self.path + 'notification.wav')  # 알람 사운드 파일 경로
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
        
    # 버튼 클릭 시 목표 위치로 이동 명령 전송
    def on_button_click(self, position):
        """버튼을 클릭하면 주어진 위치로 이동하는 명령을 전송"""
        if position in [locations["section1"], locations["section2"], locations["section3"], locations["section4"]]:
            waypoints = [locations["wp1"], locations["wp2"], position]
        else:
            waypoints = [locations["wp2"], locations["wp1"], position]
            msg = String()
            msg.data = "stop"
            self.publisher.publish(msg)

        self.node.send_goal(waypoints)
        

    def on_stop_button_click(self):
        """버튼을 클릭하면 전송한 목표를 취소하는 명령을 전송"""
        self.node.cancel_goal()
        msg = String()
        msg.data = "stop"
        self.publisher.publish(msg)

    # 텍스트 영역에 메시지 업데이트
    def update_display(self, message):
        """GUI 텍스트 창에 메시지를 추가하여 표시"""
        self.message_display.append(str(message))

    # 창 닫힐 때 자원 해제
    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        self.ros_thread.quit()
        self.ros_thread.wait()
        event.accept()


class GuiNode(Node):
    def __init__(self, GUI):
        super().__init__("gui_node")
        self.GUI = GUI
############################################################################
        self.action_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')
        self.publisher = self.create_publisher(String, 'start_tracking', 10)  # 퍼블리셔 선언

        self._goal_handle = None  # 현재 활성화된 목표 핸들을 저장하기 위한 변수


    def euler_to_quaternion(self, roll, pitch, yaw):
        # Convert Euler angles to a quaternion
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

    def send_goal(self, waypoints):
        poses = []
        for waypoint in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            yaw = waypoint[2]
            pose.pose.orientation = self.euler_to_quaternion(0, 0, yaw)
            poses.append(pose)

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = poses

        self.action_client.wait_for_server()
        self._send_goal_future = self.action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

        self.GUI.update_signal.emit(f"Moving to waypoint: x={waypoint[0]}, y={waypoint[1]}, yaw={math.degrees(yaw)} degrees")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        self.get_logger().info('Goal accepted')
        self._goal_handle = goal_handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        action_status = future.result().status
        missed_waypoints = result.missed_waypoints
        if missed_waypoints:
            self.GUI.update_signal.emit(f"Missed waypoints: {missed_waypoints}")
        else:
            self.GUI.update_signal.emit('All waypoints completed successfully!')
        
        # 로봇이 지점에 도착 시 데이터베이스에 도착 정보 저장 및 토픽 발행
        if action_status == GoalStatus.STATUS_SUCCEEDED:
            self.GUI.update_signal.emit("Goal succeeded!")
            goal_name = 'Final Goal'  # 목적지 이름을 설정해야 함. 현재는 예시로 "Final Goal"
            self.save_goal_reach_to_db(goal_name)

            # 메시지 발행
            msg = String()
            msg.data = "start"
            self.publisher.publish(msg)
            self.get_logger().info("Published success message to 'goal_status_topic'")

    def cancel_goal(self):
        if self._goal_handle is not None:
            self.get_logger().info('Attempting to cancel the goal...')
            cancel_future = self._goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        else:
            self.get_logger().info('No active goal to cancel.')

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_cancelled) > 0:
            self.get_logger().info('Goal cancellation accepted.')
            self.GUI.update_signal.emit("goal 명령을 취소합니다.")
        else:
            self.get_logger().info('Goal cancellation failed or no active goal to cancel.')
            self.GUI.update_signal.emit("goal 명령을 취소하지 못했습니다.")


def main():
    """PyQt 앱 실행을 위한 메인 함수"""
    app = QApplication(sys.argv)

    # 로그인 화면 띄우기
    login_dialog = LoginDialog()
    if login_dialog.exec_() == QDialog.Accepted:
        window = YOLOApp()
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()