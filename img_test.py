import cv2
import numpy as np
from paddleocr import PaddleOCR

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
src = cv2.imread('green_plate_009824.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = src.shape[:2]

# YOLOv5模型预处理
blob = cv2.dnn.blobFromImage(src, 1 / 255.0, (640, 640), swapRB=True, crop=False)
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

# 初始化OCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

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
image_copyed = src.copy()
# print(idxs)

# 绘制检测结果
if len(idxs) > 0:
    # 绘制检测结果
    idxs = idxs.flatten()
    yolo_detected = draw_boxes(image_copyed, boxes, confidences, class_ids, idxs, colors, classes)
    cv2.imshow("YOLO detected", yolo_detected)

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
        exit()
    yolo_cropped = src[y:y + h, x:x + w]
    # print(max_square)
    cv2.imshow("Cropped image", yolo_cropped)

    # 对图像进行处理
    # 蓝牌情况
    if class_ids[max_idx] == 0:
        # 除去蓝色通道转换成灰度图
        gray = cv2.addWeighted(yolo_cropped[:, :, 1], 0.5, yolo_cropped[:, :, 2], 0.5, 0)
        cv2.imshow("gray", gray)

        # 直方图正规化
        hist = cv2.equalizeHist(gray)
        cv2.imshow("hist", hist)

        # 阈值化
        ret, TOZERO_img = cv2.threshold(hist, 180, 255, cv2.THRESH_TOZERO)
        cv2.imshow("threshold", TOZERO_img)

        # 中值滤波
        median_filtered = cv2.medianBlur(TOZERO_img, 3)
        cv2.imshow("result", median_filtered)

        # OCR检测
        plate = ocr.ocr(TOZERO_img, det=False)
        print(plate)

        # 处理输出，判断格式
        rtn, plate_num = handle_ocr_output(plate)
        if rtn:
            print(check_plate(plate_num, label=0))


    # 绿牌情况
    elif class_ids[max_idx] == 1:
        # 取绿色通道转换成灰度图
        gray = yolo_cropped[:, :, 1]
        cv2.imshow("gray", gray)

        # 直方图正规化
        hist = cv2.equalizeHist(gray)
        cv2.imshow("hist", hist)

        # 阈值化
        ret, TRUNC_img = cv2.threshold(hist, 200, 255, cv2.THRESH_TRUNC)
        cv2.imshow("threshold", TRUNC_img)

        # 中值滤波
        median_filtered = cv2.medianBlur(TRUNC_img, 3)
        cv2.imshow("result", median_filtered)

        # OCR检测
        plate = ocr.ocr(median_filtered, det=False)
        print(plate)

        # 处理输出，判断格式
        rtn, plate_num = handle_ocr_output(plate)
        if rtn:
            print(check_plate(plate_num, label=1))

cv2.waitKey(0)
cv2.destroyAllWindows()