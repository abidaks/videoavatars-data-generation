import cv2
import argparse
import numpy as np


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--location', '-l',
		default='./mask_data',
		help='Path to save mask data')

	parser.add_argument(
		'--video', '-v',
		required=True,
		help='Video File')

	args = parser.parse_args()

	cap = cv2.VideoCapture(args.video)
	cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	flag, frame = cap.read()
	count = 0
	while flag:
		# mask = np.zeros(frame.shape[:2],np.uint8)
		# bgdModel = np.zeros((1,65),np.float64)
		# fgdModel = np.zeros((1,65),np.float64)
		# rect = (80,20,950,1000)
		# cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)
		# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		# frame = frame*mask2[:,:,np.newaxis]

		#cv2.imwrite(str(count)+".jpg", frame)

		#frame = increase_brightness(frame, value=4)

		for i in range(int(cap_width)):
			for j in range(int(cap_height)):
				if i > 1060 or (i > 630 and j > 825) or (frame[i][j][0] < 80 and frame[i][j][0] > 28 \
				and frame[i][j][1] > 75 and frame[i][j][1] < 130 \
				and frame[i][j][2] < 80 and frame[i][j][2] > 28):
					frame[i][j][0] = 0
					frame[i][j][1] = 0
					frame[i][j][2] = 0

		cv2.imwrite(args.location+"/"+str(count)+".jpg", frame)

		#cv2.imwrite(str(count)+"-2.jpg", frame)

		print(count)

		cv2.imshow("Masked-Image-Generation", frame)

		flag, frame = cap.read()
		mkey = cv2.waitKey(1)
		count = count + 1
		if mkey == ord('q'):
			flag = False
			break

		#flag = False

	cap.release()
	cv2.destroyAllWindows()
