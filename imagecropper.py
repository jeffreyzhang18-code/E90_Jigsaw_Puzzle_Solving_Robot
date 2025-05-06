import cv2
import numpy as np

def crop_bottom_left(image_path, h, w):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    bottom_left = image[h:height, 0:width-w]
    return bottom_left

img = "puzzle_1_flipped/two_pieces.jpg"
h = 100
w = 0
bottom_left_image = crop_bottom_left(img, h, w )
cv2.imwrite("puzzle_1_flipped/two_pieces_cropped.jpg", bottom_left_image)
cv2.imshow("Image", bottom_left_image)
cv2.waitKey(0)
cv2.destroyAllWindows()