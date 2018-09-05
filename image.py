import cv2
from matplotlib import pyplot as plt


def read_image(filename):
    # 이미지 파일을 읽기 위한 객체를 리턴  인자(이미지 파일 경로, 읽기 방식)
    # cv2.IMREAD_COLOR : 투명한 부분 무시되는 컬러
    # cv2.IMREAD_GRAYSCALE : 흑백 이미지로 로드
    # cv2.IMREAD_UNCHANGED : 알파 채컬을 포함한 이미지 그대로 로드
    original_image = cv2.imread(filename)
    cvt_image = cv2.cvtColor(original_image, cv2.IMREAD_GRAYSCALE)
    #cvt_image = cv2.GaussianBlur(cvt_image, (21, 21), 0)

    # plt.imshow(image)

    return cvt_image


def gradient(filename):
    img = read_image(filename)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


def contour(filename):
    img = read_image(filename)

    ret, thr = cv2.threshold(img, 127, 255, 0)
    _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))

    cnt = contours[100]

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    cv2.drawContours(img, [cnt], 0, (255, 255, 0), 1)

    plt.imshow(img)


def image_processing(filename):
    print("ㅆ발")
    frame = read_image(filename)
    print("ㅆ발")
    frame = cv2.resize(frame, (640, 480))

    print("ㅆ발")
    firstFrame = frame
    print("ㅆ발")
    frameDelta = cv2.absdiff(firstFrame, frame)
    print("ㅆ발")
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    print(thresh)
    thresh = cv2.dilate(thresh, None, iterations=2)
    print("ㅆ발")
    (_, cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(cnts)
    for c in cnts:
        print("씨벌")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_trim = frame[y:y + h, X:x + w]
        cv2.imwrite("images/" + c + ".png", img_trim)


def body(filename):

    image = cv2.imread(filename)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    body = body_cascade.detectMultiScale(grayImage, 1.01, 10)

    for (x, y, w, h) in body:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


if __name__ == "__main__":
    body("images/siba.jpeg")
