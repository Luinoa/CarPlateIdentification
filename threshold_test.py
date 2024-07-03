import cv2
import numpy as np
from collections import deque

# Get video from camera.
cap = cv2.VideoCapture(0)
ret, src = cap.read()
# Make sure the camera is open.
if not ret:
    print('Frame not captured.')
    exit()

width = cap.get(3)
height = cap.get(4)
# Set the first frame as prev_frame used to check the stability of the video.
prev_frame = src
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert the first frame to grayscale
lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# To store motion features in a period.
feas = deque()
while True:
    ret, src = cap.read()

    cv2.imshow("src", src)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Exited normally.')
        # cv2.imwrite("sharpen.jpg", sharpen)
        break

    # Check if the video is stable.
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
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