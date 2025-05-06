import cv2
import numpy as np

img1 = cv2.imread("puzzle_1_flipped/1_cropped.jpg")
img2 = cv2.imread("puzzle_1_flipped/2_cropped.jpg")
img3 = cv2.imread("puzzle_1_flipped/3_cropped.jpg")
img4 = cv2.imread("puzzle_1_flipped/4_cropped.jpg")
img5 = cv2.imread("puzzle_1_flipped/5_cropped.jpg")
img6 = cv2.imread("puzzle_1_flipped/6_cropped.jpg")

img_combined = np.hstack((img1,img2,img3,img4,img5,img6))

cv2.imwrite("puzzle_1_flipped/puzzle_1_combined.jpg", img_combined)
cv2.imshow("puzzle_1_combined", img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()