import cv2
import numpy as np
import serial

port = "COM5"
baud = 115200
ser = serial.Serial(port, baud)
ser.write(b'$X \n')
ser.write(b'$H \n')
def return_coordinates(event, x, y):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_coord = str(y-370)
        y_coord = str(x-370)
        print('Clicked at (', x_coord, ',', y_coord,')')
        command = 'G00 X' + x_coord + ' Y' + y_coord + ' \n'
        print(command)
        ser.write(command.encode('utf-8'))

window_name = 'Coordinate Plane'
size = 360
image = np.ones((size, size, 3), dtype=np.uint8) * 255

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, return_coordinates)

while True:
    cv2.imshow(window_name, image)
    if cv2.waitKey(0):
        break

cv2.destroyAllWindows()