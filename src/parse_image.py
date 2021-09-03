import cv2
import numpy as np

def extract_digit(cell):
    return []

def extract_board(img):
    modified_img = preprocess_image(img)

    # contours
    contours, _ = cv2.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse = True)
    puzzle_contour = None

    for c in contours:
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            puzzle_contour = approx
            break
    
    if puzzle_contour is None:
        raise RuntimeError("Unable to find bounding box for puzzle. Try a different image")
    
    return four_point_transform(modified_img, puzzle_contour.reshape((4,2)))
    

def preprocess_image(img):
    modified_img = cv2.GaussianBlur(img.copy(), (11,11), 0)
    modified_img = cv2.adaptiveThreshold(modified_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    modified_img = cv2.bitwise_not(modified_img)
    kernel = np.array([[0, 1, 0], [1, 1, 1],[0, 1, 0]], np.uint8)
    modified_img = cv.dilate(modified_img, kernel)
    return modified_img

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, bl, br) = rect
    
    max_width = max(int(cv.norm(tl, tr, cv.NORM_L2)),
                    int(cv.norm(bl, br, cv.NORM_L2)))
    
    max_height = max(int(cv.norm(tl, bl, cv.NORM_L2)),
                    int(cv.norm(tr, br, cv.NORM_L2)))
    
    dst = np.array([
        [0, 0],
        [max_width-1, 0],
        [max_width-1, max_height-1],
        [0, max_height-1]], dtype="float32")
    
    M = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(image, M, (max_width, max_height))


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