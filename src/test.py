from parse_image import parse_image
import matplotlib.pyplot as plt
import cv2

filename = "sudoku.jpg"
img = cv2.imread(filename, 0)
board = parse_image(img)
for row in board:
    print(row)