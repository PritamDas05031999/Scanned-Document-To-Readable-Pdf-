import urllib.request as request
import cv2
import numpy
from PIL import Image
import time
import ocrmypdf


# ip camara app output
# user_input = input("""Enter the IP address or peace w to use predetermine ip:-
# Entre your chose: """)
#
# if user_input.lower() == "w":
#     url = 'http://192.168.29.115:8080/shot.jpg?rnd=764040'
# else:
#     url = 'http://{user_input}/shot.jpg?rnd=764040'

url = 'http://192.168.29.57:8080/shot.jpg?rnd=764040'

while True:
    # ######## Image extraction ###########
    img = request.urlopen(url)
    img_bytes = bytearray(img.read())
    img_np = numpy.array(img_bytes, dtype=numpy.uint8)

    # ######## Frame dynamization ##########
    frame = cv2.imdecode(img_np, -1)
    cropped_img = frame.copy()
    height, width, _ = frame.shape
    document_contour = numpy.array([[0, 0], [width, 0], [width, height], [0, height]])
    # print(document_contour)

    # ######### Video Color reconsecration ############
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_cvt = cv2.bilateralFilter(frame_cvt, 20, 30, 30)
    frame_edge = cv2.Canny(frame_cvt, 30, 50)

    # ######### Paper Distraction Logic #############
    _, threshold = cv2.threshold(frame_edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(max_contour)
    if width == 0:
        width = 1
    if height == 1:
        height = 1
    # print(width+x, height+y)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)

    # Pixel values in the original image
    points = document_contour.reshape(4, 2)
    input_points = numpy.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[numpy.argmin(points_sum)]
    input_points[3] = points[numpy.argmax(points_sum)]

    points_diff = numpy.diff(points, axis=1)
    input_points[1] = points[numpy.argmin(points_diff)]
    input_points[2] = points[numpy.argmax(points_diff)]

    # width = width + x
    # height = height + y

    converted_points = numpy.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # print(converted_points)

    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    cropped_img = cv2.warpPerspective(cropped_img, matrix, (width, height))

    # print(matrix)

    # ######## Video Display Frame #########

    # for incise contest
    lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(150, 150))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    ling = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(ling, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # gray = cv2.bilateralFilter(gray, 20, 30, 30)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow("input", threshold)

    if cv2.waitKey(1) == ord("s"):
        img_pil = Image.fromarray(threshold)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        file_path = f"Save\DS{time_str}.pdf"
        img_pil.save(file_path)
        ocrmypdf.ocr(file_path, file_path, skip_text=True)
        print(f'save{time_str}')

    elif cv2.waitKey(1) & 0xFF == 27:
        break
