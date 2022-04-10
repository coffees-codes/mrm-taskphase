import cv2
import numpy as np

load_from_sys = True

if load_from_sys:
	hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5,5), np.uint8)

canvas = None

x1 = 0
y1 = 0

noise_thresh = 800

while True:
	# tuple?
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)

	if canvas is None:
		canvas = np.zeros_like(frame)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if load_from_sys:
		#np.array([x, y, z])
		lower_range = hsv_value[0]
		upper_range = hsv_value[1]

	# mask = cv2.inRange(frame, lower_range, upper_range)
	mask = cv2.inRange(hsv, lower_range, upper_range)


	mask = cv2.erode(mask, kernel, iterations=1)
	mask = cv2.dilate(mask, kernel, iterations=2)

	contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noise_thresh:
		c = max(contours, key = cv2.contourArea)
		x2, y2, w, h = cv2.boundingRect(c)

		if x1 == 0 and y1 == 0:
			x1, y1 = x2, y2
		else:
			canvas = cv2.line(canvas, (x1, y1), (x2, y2), [0, 255, 255], 4)

		x1, y1 = x2, y2
	
	else:
		x1, y1 = 0, 0

	frame = cv2.add(frame, canvas)

	stacked = np.hstack((canvas, frame))
	cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx = 0.6, fy = 0.6))

	if cv2.waitKey(1) == 13:
		break

	#Clear the canvas when 'c' is pressed
	if cv2.waitKey(1) & 0xFF == ord('c'):
		canvas = None

	if cv2.waitKey(1) & 0xFF == ord('s'):
		cv2.imwrite('image.jpg', canvas)

		# READING THE IMAGE IN GRAYSCALE
		grayscale_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

		h, w = grayscale_image.shape
		print("Original Height and Width:", h, "x", w)

		# DILATING THE IMAGE
		kernel = np.ones((5, 5), np.uint8)
		mask = cv2.dilate(grayscale_image, kernel, iterations=10)

		# RESIZING THE IMAGE IN GRAYSCALE
		down_width = 28
		down_height = 28
		down_points = (down_width, down_height)
		resize_down = cv2.resize(mask, down_points, interpolation=cv2.INTER_LINEAR)
		h, w = resize_down.shape
		print("New Height and Width:", h, "x", w)

		# SAVING THE RESIZED IMAGE
		cv2.imwrite('final_image.jpg', resize_down)

cv2.destroyAllWindows()
cap.release()