import cv2
import numpy as np
from collections import deque
from torchvision import transforms


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
        for i in range(len(class_iou)):
            confidence = box_iou * class_iou[i]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([width / 640, height / 640, width / 640, height / 640])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(i)

    # 应用非极大值抑制
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    image = src.copy()
    # 绘制检测结果
    if len(idxs) > 0:
        idxs = idxs.flatten()
        image = draw_boxes(src, boxes, confidences, class_ids, idxs, colors, classes)

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
    for i in range(len(class_iou)):
        confidence = box_iou * class_iou[i]
        if confidence > conf_threshold:
            box = detection[0:4] * np.array([width / 640, height / 640, width / 640, height / 640])
            (centerX, centerY, w, h) = box.astype("int")
            x = int(centerX - (w / 2))
            y = int(centerY - (h / 2))
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(i)

# 应用非极大值抑制
idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# 绘制检测结果
if len(idxs) > 0:
    idxs = idxs.flatten()
    image = draw_boxes(image, boxes, confidences, class_ids, idxs, colors, classes)

# 显示结果图像
cv2.imshow('YOLOv5 Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


