import cv2
import numpy as np
from skimage.segmentation import clear_border
import pytesseract
import re
# import matplotlib.pyplot as plt

def parse_image(img):
    board_img = extract_board(img)
    # assume we are working on a 9x9 sudoku board with equal sized cells
    strideX = board_img.shape[1] // 9
    strideY = board_img.shape[0] // 9
    board_array = []
    # step through each cell in sudoku board image
    for i in range(0,9):
        row = []
        for j in range(0, 9):
            cell = board_img[strideY*i:strideY*(i+1), strideX*j:strideX*(j+1)]
            digit = extract_digit_image(cell)
            if (digit is None):
                row.append("_")
            else:
                row.append(parse_digit(digit))
        board_array.append(row)
    return board_array

def parse_digit(digit):
    extract_digit = r'[0123456789]'
    # TODO: reconsider using tesseract
    tesseract_result = pytesseract.image_to_string(digit, config = '--psm 6')
    nums = re.findall(extract_digit, tesseract_result)
    if len(nums) > 0:
        return int(nums[0])
    return "_"
    

def extract_digit_image(cell):
    # use sklearn function to clean up cell border a bit
    working_cell = clear_border(cell)
    cnts, _ = cv2.findContours(working_cell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None

    # find the biggest contour and if it exceeds a certain portion of cell return that contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(working_cell.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    (h, w) = working_cell.shape
    percent_filled = cv2.countNonZero(mask) / float(w*h)

    if percent_filled < 0.07:
        return None

    digit = cv2.bitwise_and(working_cell, working_cell, mask=mask)
    return digit

def extract_board(img):
    modified_img = preprocess_image(img)

    # trace out all contours
    contours, _ = cv2.findContours(modified_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # have largest contour first in list
    contours = sorted(contours, key=cv2.contourArea, reverse = True)
    # find the largest 4 sided contour
    puzzle_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            puzzle_contour = approx
            break
    
    if puzzle_contour is None:
        raise RuntimeError("Unable to find bounding box for puzzle. Try a different image")
    
    return four_point_transform(modified_img, puzzle_contour.reshape((4,2)))
    

def preprocess_image(img):
    # apply blur to get rid of noise
    modified_img = cv2.GaussianBlur(img.copy(), (11,11), 0)
    # apply a threshold to only "pick up" actual lines
    modified_img = cv2.adaptiveThreshold(modified_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # invert image as whites are typically used as lines
    modified_img = cv2.bitwise_not(modified_img)
    # dilate image to fill in cracks in lines caused by thresholding
    kernel = np.array([[0, 1, 0], [1, 1, 1],[0, 1, 0]], np.uint8)
    modified_img = cv2.dilate(modified_img, kernel)
    return modified_img

def four_point_transform(image, pts):
    # give bird's eye view of image
    rect = order_points(pts)
    (tl, tr, bl, br) = rect
    
    # find max width and height using the straightline distances between corners
    max_width = max(int(cv2.norm(tl, tr, cv2.NORM_L2)),
                    int(cv2.norm(bl, br, cv2.NORM_L2)))
    
    max_height = max(int(cv2.norm(tl, bl, cv2.NORM_L2)),
                    int(cv2.norm(tr, br, cv2.NORM_L2)))
    
    dst = np.array([
        [0, 0],
        [max_width-1, 0],
        [max_width-1, max_height-1],
        [0, max_height-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def order_points(pts):
    # points will be ordered clockwise starting from top left 
    rect = np.zeros((4,2), dtype="float32")
    
    # top left will have smallest sum
    # bottom right will have largest sum
    point_sums = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(point_sums)]
    rect[2] = pts[np.argmax(point_sums)]
    
    # top right will have smallest difference
    # bottom left will have largest difference
    point_diffs = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(point_diffs)]
    rect[3] = pts[np.argmax(point_diffs)]
    
    return rect