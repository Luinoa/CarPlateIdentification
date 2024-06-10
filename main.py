import cv2
import numpy as np
from collections import deque
# Read into gray image
src = cv2.imread("01-90_89-139&598_334&667-328&664_143&660_142&600_327&604-0_0_16_32_31_27_24-61-13.jpg",
                 cv2.IMREAD_COLOR)


# Get video from camera.
cap = cv2.VideoCapture(0)

ret, src = cap.read()

# Make sure the camera is open.
if not ret:
    print('Frame not captured.')
    exit()

# Set the first frame as prev_frame used to check the stability of the video.
prev_frame = src
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert the first frame to grayscale
lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# To store motion features in a period.
feas = deque()

# Read into frames.
while True:
    ret, src = cap.read()

    # Make sure the frame is input.
    if not ret:
        print('Frame not captured.')
        break

    # Denoise and sharpen the frame.
    gaussian = cv2.GaussianBlur(src, (5, 5), 0)
    sharpen_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
    sharpen =cv2.filter2D(gaussian, -1, sharpen_kernel)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Exited normally.')
        #cv2.imwrite("sharpen.jpg", sharpen)
        break
    # print(cap.get(3), cap.get(4)) # Get width and height.

    # Display the videos.
    cv2.imshow("src", src)
    cv2.imshow("gaussain", gaussian)
    cv2.imshow("sharpen", sharpen)

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

    if fea_period < 10:
        print('Stable')
    else:
        print('Unstable')



cap.release()

"""
# These are the codes to read images.
gaussian = cv2.GaussianBlur(src, (5, 5), 0)
sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
sharpen_image =cv2.filter2D(src, -1, sharpen_kernel)

cv2.imshow("src",src)
cv2.imshow("gaussain", gaussian)
cv2.imshow("sharpen", sharpen_image)
cv2.waitKey(0)
"""

cv2.destroyAllWindows()