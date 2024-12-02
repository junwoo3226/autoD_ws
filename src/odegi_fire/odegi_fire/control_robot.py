import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import os
from ultralytics import YOLO
import numpy as np

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


# def process_frame(annotated_frame, results, pub):
#     """
#     탐지된 객체의 바운딩 박스를 기반으로 구역에 따라 TurtleBot3 제어 메시지 발행.
#     """
#     # 화면 중앙 기준 설정
#     frame_width = annotated_frame.shape[1]
#     left_boundary = frame_width // 3
#     right_boundary = 2 * frame_width // 3

#     target_linear_velocity = 0.0
#     target_angular_velocity = 0.0

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf[0]
#             cls = int(box.cls)
#             class_name = result.names[cls]

#             # 바운딩 박스와 라벨 그리기
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # 탐지된 객체의 중심 좌표 계산
#             cx = (x1 + x2) // 2

#             # 탐지된 객체가 어느 구역에 있는지 판별
#             if conf > 0.75:
#                 if cx < left_boundary:  # 왼쪽
#                     target_angular_velocity = check_angular_limit_velocity(
#                         target_angular_velocity + ANG_VEL_STEP_SIZE
#                     )
#                     print(f"왼쪽에 {class_name} 감지됨! Angular Velocity Increased: {target_angular_velocity}")
#                 elif cx > right_boundary:  # 오른쪽
#                     target_angular_velocity = check_angular_limit_velocity(
#                         target_angular_velocity - ANG_VEL_STEP_SIZE
#                     )
#                     print(f"오른쪽에 {class_name} 감지됨! Angular Velocity Decreased: {target_angular_velocity}")
#                 else:  # 중앙
#                     target_linear_velocity = 0.0
#                     target_angular_velocity = 0.0
#                     print(f"중앙에 {class_name} 감지됨! Stopped")

#                 # Twist 메시지 생성 및 발행
#                 twist = Twist()
#                 twist.linear.x = target_linear_velocity
#                 twist.angular.z = target_angular_velocity
#                 pub.publish(twist)
#                 return


class TurtleBot3Controller(Node):
    def __init__(self):
        super().__init__('turtlebot3_controller')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("TurtleBot3 controller node started.")

    def publish_velocity(self, annotated_frame, results):
        process_frame(annotated_frame, results, self.publisher_)


# def main():
#     rclpy.init()
#     node = TurtleBot3Controller()

#     # OpenCV 및 YOLO 설정
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     model = YOLO('/home/rokey8/rokey8_D5_ws/src/fire/fire/best.pt')

#     try:
#         while rclpy.ok():
#             ret, frame = cap.read()
#             if not ret:
#                 node.get_logger().warn("Failed to capture frame from webcam.")
#                 break

#             # YOLO 추론 및 결과 처리
#             results = model.predict(source=frame, conf=0.5, save=False, verbose=False)
#             annotated_frame = results[0].plot()

#             # 속도 메시지 발행
#             node.publish_velocity(annotated_frame, results)

#             # OpenCV 창에 결과 표시
#             cv2.imshow("TurtleBot3 YOLO Detection", annotated_frame)

#             # 종료 조건
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()


def process_frame(annotated_frame, results, pub):
    """
    YOLO Tracking 결과를 기반으로 객체의 ID와 이름을 표시하며 TurtleBot3를 제어.
    """
    # 화면 중앙 기준 설정
    frame_width = annotated_frame.shape[1]
    left_boundary = frame_width // 3
    right_boundary = 2 * frame_width // 3

    target_linear_velocity = 0.0
    target_angular_velocity = 0.0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            conf = box.conf[0]  # 신뢰도
            cls = int(box.cls)  # 클래스 ID
            class_name = result.names[cls] if cls in result.names else "Unknown"  # 클래스 이름
            obj_id = int(box.id[0]) if box.id is not None else -1  # 트래킹 ID

            # ID와 이름을 결합한 라벨 생성
            label = f"ID: {obj_id} {class_name}"  # 예: ID: 1 fire

            # 바운딩 박스와 라벨 그리기
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 탐지된 객체의 중심 좌표 계산
            cx = (x1 + x2) // 2

            # 탐지된 객체가 어느 구역에 있는지 판별
            if conf > 0.75:
                if cx < left_boundary:  # 왼쪽
                    target_angular_velocity = check_angular_limit_velocity(
                        target_angular_velocity + ANG_VEL_STEP_SIZE
                    )
                    print(f"왼쪽에 {label} 감지됨! Angular Velocity Increased: {target_angular_velocity}")
                elif cx > right_boundary:  # 오른쪽
                    target_angular_velocity = check_angular_limit_velocity(
                        target_angular_velocity - ANG_VEL_STEP_SIZE
                    )
                    print(f"오른쪽에 {label} 감지됨! Angular Velocity Decreased: {target_angular_velocity}")
                else:  # 중앙
                    target_linear_velocity = 0.0
                    target_angular_velocity = 0.0
                    print(f"중앙에 {label} 감지됨! Stopped")

                # Twist 메시지 생성 및 발행
                twist = Twist()
                twist.linear.x = target_linear_velocity
                twist.angular.z = target_angular_velocity
                pub.publish(twist)


def main():
    rclpy.init()
    node = TurtleBot3Controller()

    # OpenCV 및 YOLO 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model = YOLO('/home/rokey8/rokey8_D5_ws/src/fire/fire/best.pt')

    try:
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret:
                node.get_logger().warn("Failed to capture frame from webcam.")
                break

            # YOLO Tracking 추론
            results = model.track(source=frame, conf=0.5, persist=True, verbose=False)

            # 추론 결과 처리 및 속도 메시지 발행
            annotated_frame = results[0].plot()
            node.publish_velocity(annotated_frame, results)

            # OpenCV 창에 결과 표시
            cv2.imshow("TurtleBot3 YOLO Tracking", annotated_frame)

            # 종료 조건
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
