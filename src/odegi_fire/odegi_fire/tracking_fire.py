import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import threading
import queue

# TurtleBot3 속도 제한 값
BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84
LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

# 모델 환경 변수
TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')

def constrain(input_vel, low_bound, high_bound):
    """속도 제한 함수"""
    return max(low_bound, min(input_vel, high_bound))

def check_linear_limit_velocity(velocity):
    """선속도 제한"""
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)

def check_angular_limit_velocity(velocity):
    """각속도 제한"""
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)


def process_frame(annotated_frame, results, pub):
    """탐지된 객체의 바운딩 박스를 기반으로 구역에 따라 TurtleBot3 제어 메시지 발행"""
    frame_width = annotated_frame.shape[1]
    left_boundary = frame_width // 3
    right_boundary = 2 * frame_width // 3

    target_linear_velocity = 0.0
    target_angular_velocity = 0.0

    for box, track_id in zip(results[0].boxes.xywh.cpu().numpy(), 
                             results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []):
        x, y, w, h = box  # 중심 x, y, 폭, 높이
        cx = int(x)  # 중심 좌표
        conf = results[0].boxes.conf.cpu().numpy()[0]  # confidence 값

        if conf > 0.5:  # 신뢰도 필터
            if cx < left_boundary:  # 왼쪽
                target_angular_velocity = check_angular_limit_velocity(
                    target_angular_velocity + ANG_VEL_STEP_SIZE
                )
                print(f"왼쪽에 객체(ID={track_id}) 감지됨! Angular Velocity Increased: {target_angular_velocity}")
            elif cx > right_boundary:  # 오른쪽
                target_angular_velocity = check_angular_limit_velocity(
                    target_angular_velocity - ANG_VEL_STEP_SIZE
                )
                print(f"오른쪽에 객체(ID={track_id}) 감지됨! Angular Velocity Decreased: {target_angular_velocity}")
            else:  # 중앙
                target_linear_velocity = 0.0
                target_angular_velocity = 0.0
                print(f"중앙에 객체(ID={track_id}) 감지됨! Stopped")

            # Twist 메시지 생성 및 발행
            twist = Twist()
            twist.linear.x = target_linear_velocity
            twist.angular.z = target_angular_velocity
            pub.publish(twist)
            return

def capture_frames(cap, frame_queue):
    while rclpy.ok():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        if not frame_queue.full():
            frame_queue.put(frame)

def process_frames(frame_queue, result_queue, model):
    while rclpy.ok():
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model.track(source=frame, persist=True, conf=0.5)
            
            # results = model.track(source=cv2.resize(frame, (160, 120)), persist=True, conf=0.5)

            if not result_queue.full():
                result_queue.put((frame, results))

def publish_results(result_queue, node, track_history):
    while rclpy.ok():
        if not result_queue.empty():
            frame, results = result_queue.get()
            if results and results[0].boxes:
                annotated_frame = results[0].plot()

                # 추적 결과 시각화
                for box, track_id in zip(results[0].boxes.xywh.cpu().numpy(), 
                                         results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []):
                    x, y, w, h = box
                    track_history[track_id].append((int(x), int(y)))

                    # 트랙 선 시각화
                    points = np.array(track_history[track_id], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 127, 255), thickness=2)

                # 속도 메시지 발행
                node.publish_velocity(annotated_frame, results)

                cv2.imshow("TurtleBot3 YOLO Tracking", annotated_frame)
            else:
                node.get_logger().info("No objects detected in this frame.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



class TurtleBot3Controller(Node):
    def __init__(self):
        super().__init__('turtlebot3_controller')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("TurtleBot3 controller node started.")

    def publish_velocity(self, annotated_frame, results):
        process_frame(annotated_frame, results, self.publisher_)



def main():
    rclpy.init()
    node = TurtleBot3Controller()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 제한

    model = YOLO('/home/rokey8/rokey8_D5_ws/src/fire/fire/best.pt')
    track_history = defaultdict(lambda: [])

    frame_queue = queue.Queue(maxsize=5)
    result_queue = queue.Queue(maxsize=5)

    # 스레드 생성
    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue), daemon=True)
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, result_queue, model), daemon=True)
    publish_thread = threading.Thread(target=publish_results, args=(result_queue, node, track_history), daemon=True)

    # 스레드 생성 여부 확인
    print("스레드 생성 여부 확인")
    print(f"Capture thread alive: {capture_thread.is_alive()}")
    print(f"Process thread alive: {process_thread.is_alive()}")
    print(f"Publish thread alive: {publish_thread.is_alive()}")

    print("--스레드 생성 여부 확인--")


    # 스레드 시작
    # capture_thread.start()
    # process_thread.start()
    # publish_thread.start()
    print("Starting capture thread...")
    capture_thread.start()

    print("Starting process thread...")
    process_thread.start()

    print("Starting publish thread...")
    publish_thread.start()


    print("스레드 생성 여부 확인")
    print(f"Capture thread alive: {capture_thread.is_alive()}")
    print(f"Process thread alive: {process_thread.is_alive()}")
    print(f"Publish thread alive: {publish_thread.is_alive()}")

    print("--스레드 생성 여부 확인--")
    # ROS 노드 유지
    try:
        # rclpy.spin(node)
        # cap.read()
        print("Spinning ROS node...")
        rclpy.spin(node)
        print("ROS node shut down.")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()