# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
def sort_contours(cnts, method="left-to-right"):
# initialize the reverse flag and sort index
    reverse = False
    i = 0
# handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-totop":
        reverse = True
# handle if we are sorting against the y-coordinate rather than
# the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottomto-top":
        i = 1
# construct the list of bounding boxes and sort them from top to
# bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
# return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
def draw_contour(image, c, i):
# compute the center of the contour area and draw a circle
# representing the center
    M = cv2.moments(c)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the countour number on the image
        #cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY),
        #cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        print("無，跳過")
# return the image with the contour number drawn on it
    return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
ap.add_argument("-m", "--method", required=True, help="Sorting method")
#ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
# load the image image, convert it to grayscale, and detect edges
for imagePath in glob.glob(args["images"] + "/*.jpg"):
    # Load the image
    image = cv2.imread(imagePath)

    # Convert the image to grayscale, blur it, and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    print("找到的輪廓數量:", len(cnts))

    # Sort the contours
    (cnts, boundingBoxes) = sort_contours(cnts, method=args["method"])
    # 創建一個列表來存儲近似多邊形和對應的面積
    approx_with_area = []

    # Draw the contours on the original image
    for (i, c) in enumerate(cnts):
        # Approximate the contour to a polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        print("original: {}, approx: {}".format(len(c), len(approx)))
        area = cv2.contourArea(approx)
        approx_with_area.append((approx, area))
        # 篩選出點數小於 10 的近似多邊形和面積
        small_approx_with_area = [item for item in approx_with_area if len(item[0]) < 10]

        # 根據面積排序 (降序)
        small_approx_with_area = sorted(small_approx_with_area, key=lambda x: x[1], reverse=True)[:3]
    for i, (approx, area) in enumerate(small_approx_with_area):
        if len(approx) < 10:
            # Draw the polygon on the image
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            # 計算輪廓的中心點
            M = cv2.moments(approx)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 在中心點處標示編號
                cv2.putText(image, "#{}".format(i + 1), (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:
                print("沒有中心點")
            # Call the draw_contour function for additional visualizations
            draw_contour(image, c, i)

    # Show the image with contours
    cv2.imshow("Image", image)
    cv2.waitKey(0)
