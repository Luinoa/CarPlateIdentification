import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from collections import deque

province_list = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

letter_list = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"]

number_list = [
    "0", "1","2", "3", "4",
    "5", "6", "7", "8", "9"
]


def check_plate(plate_num, label):
    # 确认长度
    # 蓝牌
    if label == 0:
        if len(plate_num) != 8:
            return False

    # 绿牌
    if label == 1:
        if len(plate_num) != 9:
            return False

    # 确认车牌字符是否在对应字符集里
    if plate_num[0] not in province_list:
        return False
    if plate_num[1] not in letter_list:
        return False
    if plate_num[2] != '·':
        return False
    car_num = plate_num[3:]
    for char_in_plate in car_num:
        if char_in_plate not in letter_list and char_in_plate not in number_list:
            return False

    return True

# 取出IoU最大的识别结果
def handle_ocr_output(plate):
    IoU = 0.0
    valid_IoU = 0.9
    plate_num = None
    rtn = False
    for item in plate:
        if item is None:
            continue
        if item[0][1] < valid_IoU:
            continue
        if item[0][1] > IoU:
            plate_num = item[0][0]
            rtn = True
    return rtn, plate_num



def draw_boxes(image, boxes, confidences, class_ids, idxs, colors, classes):
    for i in idxs:
        x, y, w, h = boxes[i]
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# 加载YOLOv5的ONNX模型
net = cv2.dnn.readNetFromONNX('best.onnx')

# Get video from camera.
cap = cv2.VideoCapture(0)

ret, src = cap.read()

# Make sure the camera is open.
if not ret:
    print('Frame not captured.')
    exit()

width = cap.get(3)
height = cap.get(4)

# 初始化颜色和类别名
colors = np.array([
    [255, 0, 0],
    [0, 255, 0]
])
classes = ['blue_plate', 'green_plate']

# 初始化OCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

# Read into frames.
while True:
    ret, src = cap.read()

    # Make sure the frame is input.
    if not ret:
        print('Frame not captured.')
        break

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Exited normally.')
        #cv2.imwrite("sharpen.jpg", sharpen)
        break
    # print(cap.get(3), cap.get(4)) # Get width and height.

    # Display the videos.
    cv2.imshow("src", src)

    # YOLOv5模型预处理
    blob = cv2.dnn.blobFromImage(src, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    # 进行前向传播，获取输出
    outputs = net.forward()
    # print(outputs.shape)

    # YOLOv5参数
    conf_threshold = 0.5
    nms_threshold = 0.4

    # 解析输出
    boxes = []
    confidences = []
    class_ids = []

    # 将需要的box提取出来
    for detection in outputs[0]:
        box_iou = detection[4]
        class_iou = detection[5:]
        class_id = 0
        for i in range(len(class_iou)):
            if class_iou[i] > class_iou[class_id]:
                class_id = i
        confidence = box_iou * class_iou[class_id]
        if confidence > conf_threshold:
            box = detection[0:4] * np.array([width / 640, height / 640, width / 640, height / 640])
            (centerX, centerY, w, h) = box.astype("int")
            x = int(centerX - (w / 2))
            y = int(centerY - (h / 2))
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)


    # 应用非极大值抑制
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    image = src.copy()
    # 绘制检测结果
    if len(idxs) > 0:
        # 绘制检测结果
        idxs = idxs.flatten()
        yolo_detected = draw_boxes(image, boxes, confidences, class_ids, idxs, colors, classes)

        # 裁剪图像
        max_square = 0
        max_idx = idxs[0]

        for idx in idxs:
            square = boxes[idx][2] * boxes[idx][3]
            if square > max_square:
                max_square = square
                max_idx = idx

        x, y, w, h = boxes[max_idx]
        if x <= 0 or y <= 0 or w <= 0 or h <= 0:
            continue
        yolo_cropped = src[y:y + h, x:x + w]
        # print(max_square)
        # cv2.imshow("Cropped image", yolo_cropped)

        # 对图像进行处理
        # 蓝牌情况
        if class_ids[idx] == 0:
            # 除去蓝色通道转换成灰度图
            gray = cv2.addWeighted(yolo_cropped[:, :, 1], 0.5, yolo_cropped[:, :, 2], 0.5, 0)
            # cv2.imshow("Gray", gray)

            # 直方图正规化
            hist = cv2.equalizeHist(gray)
            # cv2.imshow("hist", hist)

            # 阈值化
            ret, TOZERO_img = cv2.threshold(hist, 180, 255, cv2.THRESH_TOZERO)
            cv2.imshow("result", TOZERO_img)

            # OCR检测
            plate = ocr.ocr(TOZERO_img, det=False)
            print(plate)

            # 处理输出，判断格式
            rtn, plate_num = handle_ocr_output(plate)
            if rtn:
                print(check_plate(plate_num, label=0))


        # 绿牌情况
        elif class_ids[idx] == 1:
            # 除去绿色通道转换成灰度图
            gray = cv2.addWeighted(yolo_cropped[:, :, 0], 0.5, yolo_cropped[:, :, 2], 0.5, 0)
            # cv2.imshow("Gray", gray)

            # 直方图正规化
            hist = cv2.equalizeHist(gray)
            # cv2.imshow("hist", hist)

            # 阈值化
            ret, TRUNC_img = cv2.threshold(hist, 150, 255, cv2.THRESH_TRUNC)
            cv2.imshow("threshold", TRUNC_img)

            # 中值滤波
            median_filtered = cv2.medianBlur(TRUNC_img, 3)
            # cv2.imshow("result", median_filtered)

            # OCR检测
            plate = ocr.ocr(median_filtered, det=False)
            print(plate)

            # 处理输出，判断格式
            rtn, plate_num = handle_ocr_output(plate)
            if rtn:
                print(check_plate(plate_num, label=1))


    cv2.imshow('YOLO detected', image)


cv2.destroyAllWindows()

"""
# Set the first frame as prev_frame used to check the stability of the video.
prev_frame = src
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert the first frame to grayscale
lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# To store motion features in a period.
feas = deque()

    # Check if the video is stable.
    gray = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray,
                                 maxCorners=100,
                                 qualityLevel=0.3,
                                 minDistance=7,
                                 blockSize=7)  # Detect good features to track in the previous frame

    if p0 is None:
        prev_gray = gray.copy()
        continue

    #  Calculate optical flow to get the new feature positions in the current frame.
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                           winSize=(15, 15),
                                           maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = gray.copy()  # Update the previous frame to the current frame

    if p1 is None:
        continue

    motion = p1 - p0  # Calculate the displacement of the feature points (motion vectors)

    # Calculate the feature value of the motion.
    abs_sum = 0
    for i in range(motion.shape[0]):
        abs_sum = abs_sum + abs(motion[i, 0, 0]) + abs(motion[i, 0, 1])
    fea_val = abs_sum / motion.shape[0]

    # Store the feature value.
    if len(feas) > 10:
        feas.popleft()
    feas.append(fea_val)

    fea_period = sum(feas) / len(feas)
    print(fea_period, end=' ')

    # Set a threshold.
    if fea_period < 10:
        print('Stable')
    else:
        print('Unstable')



cap.release()

"""


"""
# 测试图片用代码
def draw_boxes(image, boxes, confidences, class_ids, idxs, colors, classes):
    for i in idxs:
        x, y, w, h = boxes[i]
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


# 加载YOLOv5的ONNX模型
net = cv2.dnn.readNetFromONNX('best.onnx')

# 读取输入图像
image = cv2.imread('6d86b00da904752acef7bb6d66d86e7.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

# YOLOv5模型预处理
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)

# 进行前向传播，获取输出
outputs = net.forward()
# print(outputs.shape)

# YOLOv5参数
conf_threshold = 0.5
nms_threshold = 0.4

# 初始化颜色和类别名
colors = np.array([
    [255, 0, 0],
    [0, 255, 0]
])
classes = ['blue_plate', 'green_plate']

# 解析输出
boxes = []
confidences = []
class_ids = []

# print(outputs.shape)
# print(outputs[0, 0])

# 将需要的box提取出来
for detection in outputs[0]:
    box_iou = detection[4]
    class_iou = detection[5:]
    class_id = 0
    for i in range(len(class_iou)):
        if class_iou[i] > class_iou[class_id]:
            class_id = i
    confidence = box_iou * class_iou[class_id]
    if confidence > conf_threshold:
        box = detection[0:4] * np.array([width / 640, height / 640, width / 640, height / 640])
        (centerX, centerY, w, h) = box.astype("int")
        x = int(centerX - (w / 2))
        y = int(centerY - (h / 2))
        boxes.append([x, y, int(w), int(h)])
        confidences.append(float(confidence))
        class_ids.append(class_id)

# 应用非极大值抑制
idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
image_copyed = image.copy()
# print(idxs)

if len(idxs) > 0:
    # 绘制检测结果
    idxs = idxs.flatten()
    yolo_detected = draw_boxes(image_copyed, boxes, confidences, class_ids, idxs, colors, classes)

    # 裁剪图像
    max_square = 0
    max_idx = idxs[0]

    for idx in idxs:
        print(boxes[idx])
        square = boxes[idx][2] * boxes[idx][3]
        if square > max_square:
            max_square = square
            max_idx = idx

    x, y, w, h = boxes[max_idx]
    yolo_cropped = image[y:y + h, x:x + w]
    cv2.imshow("Cropped image", yolo_cropped)

    # 对图像进行处理
    # 转换为灰度图像
    gray = cv2.cvtColor(yolo_cropped, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    median_filtered = cv2.medianBlur(gray, 5)

    # 中值滤波
    cv2.imshow("Median Blured", median_filtered)

    # 阈值化
    ret, OTSU_img = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("threshold", OTSU_img)

    # 锐化
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(OTSU_img, -1, kernel)
    cv2.imshow("Sharpen", sharpened_image)


# 显示结果图像
# cv2.imshow('YOLOv5 Detection', yolo_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



