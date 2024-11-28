# 캠이 켜지는 지 확인
import cv2

def main():
    # 웹캠 열기 (0은 기본 웹캠, 외부 카메라는 1, 2 등의 번호로 변경)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("ESC 키를 누르면 프로그램이 종료됩니다.")
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 프레임 출력
        cv2.imshow('Camera Test', frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) == 27:  # 27은 ESC 키의 ASCII 코드
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
