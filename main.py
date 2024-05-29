import cv2

# Get video from camera.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("The camera cannot be opened.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print('Frame not captured.')
        break

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Exited normally.')
        break
    cv2.imshow('Camera', frame)
cap.release()
cv2.destroyAllWindows()