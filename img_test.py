import cv2
import numpy as np

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