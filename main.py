import cv2
import numpy as np
from paddleocr import PaddleOCR
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
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

def calculate_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    return cdf_normalized

def histogram_specification(src_image, target_hist):
    # 计算图像的直方图
    src_hist = cv2.calcHist([src_image], [0], None, [256], [0, 256]).flatten()

    # 计算源图像的CDF
    src_cdf = calculate_cdf(src_hist)

    # 计算目标直方图的CDF
    target_cdf = calculate_cdf(target_hist)

    # 创建映射表
    mapping = np.zeros(256, dtype=np.uint8)

    src_value = 0
    for target_value in range(256):
        while src_value < 256 and src_cdf[src_value] < target_cdf[target_value]:
            src_value += 1
        mapping[target_value] = src_value if src_value < 256 else 255

    # 映射源图像
    specified_image = cv2.LUT(src_image, mapping)

    return specified_image

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
            IoU = item[0][1]
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

# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# 初始化OCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

hist_template = cv2.imread("hist_template.png", cv2.IMREAD_GRAYSCALE)
target_hist = cv2.calcHist([hist_template], [0], None, [256], [0, 256]).flatten()

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

    # 测试没有处理前OCR的能力
    # print(ocr.ocr(src))

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
        cv2.imshow('YOLO detected', yolo_detected)

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
        print(max_square)
        # cv2.imshow("Cropped image", yolo_cropped)

        # 对图像进行处理
        # 蓝牌情况
        if class_ids[max_idx] == 0:
            # 除去蓝色通道转换成灰度图
            gray = cv2.addWeighted(yolo_cropped[:, :, 1], 0.5, yolo_cropped[:, :, 2], 0.5, 0)
            cv2.imshow("Gray", gray)

            # 自适应直方图均衡化
            hist = clahe.apply(gray)
            cv2.imshow("hist", hist)

            # 中值滤波
            median_filtered = cv2.medianBlur(hist, 3)
            cv2.imshow("result", median_filtered)

            # OCR检测
            plate = ocr.ocr(hist, det=False)
            print(plate)

            # 处理输出，判断格式
            rtn, plate_num = handle_ocr_output(plate)
            if rtn:
                print(check_plate(plate_num, label=0))


        # 绿牌情况
        elif class_ids[max_idx] == 1:
            # 取绿色通道转换成灰度图
            gray = yolo_cropped[:, :, 1]
            cv2.imshow("Gray", gray)

            # 自适应直方图均衡化
            hist = clahe.apply(gray)
            cv2.imshow("hist", hist)

            # 中值滤波
            median_filtered = cv2.medianBlur(hist, 3)
            cv2.imshow("result", median_filtered)

            # OCR检测
            plate = ocr.ocr(median_filtered, det=False)
            print(plate)

            # 处理输出，判断格式
            rtn, plate_num = handle_ocr_output(plate)
            if rtn:
                print(check_plate(plate_num, label=1))



cv2.destroyAllWindows()


