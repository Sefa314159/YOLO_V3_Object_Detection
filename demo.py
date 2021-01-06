"""Demo for use yolo v3
"""
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
# -------------------------------------------------------------------------------------
def process_image(img):
    """
    Resize, reduce and expand image.
    
    # Argument:
        img: original image.
        
    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image  = cv2.resize(img, (416, 416), interpolation = cv2.INTER_CUBIC)
    image  = np.array(image, dtype = 'float32')
    image /= 255.
    image  = np.expand_dims(image, axis = 0)

    return image
# -------------------------------------------------------------------------------------
def get_classes(file):
    """
    Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.
    """
    with open(file) as f:
        class_names = f.readlines()
        
    class_names = [c.strip() for c in class_names]

    return class_names
# -------------------------------------------------------------------------------------
def generate_colors(class_names):
        hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
        colors     = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors     = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(84)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        return colors
# -------------------------------------------------------------------------------------
def draw(image, boxes, scores, classes, all_classes, colors):
    
    """
    Draw the boxes on the image.
    
    # Argument:
        image       : original image.
        boxes       : ndarray, boxes of objects.
        classes     : ndarray, classes of objects.
        scores      : ndarray, scores of objects.
        all_classes : all classes name.
    """
    
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        top        = max(0, np.floor(x + 0.5).astype(int))
        left       = max(0, np.floor(y + 0.5).astype(int))
        right      = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom     = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        thickn     = math.ceil((w + h) / 200)
        cv2.rectangle(image, 
                      (top, left), 
                      (right, bottom), 
                      color     = colors[cl], 
                      thickness = thickn)
        # cv2.putText(image, 
        #             text      = f"{all_classes[cl]} {score:.2f}",
        #             org       = (top + 5, left + 5), 
        #             fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale = 0.7, 
        #             color     = (255, 255, 255), 
        #             thickness = thickn, 
        #             lineType  = cv2.LINE_AA)

        print(f'class               : {all_classes[cl]}')
        print(f'accuracy            : % {(score * 100):.2f}')
        print(f'bounding box x1, y1 : {box[0]:.2f}, {box[1]:.2f}')
        print(f'bounding box w, h   : {box[2]:.2f}, {box[3]:.2f}')
        print(f'bounding box x2, y2 : {(box[0] + box[2]):.2f}, {(box[1] + box[3]):.2f}\n')
# -------------------------------------------------------------------------------------
def detect_image(image, yolo, all_classes):
    
    """
    Use yolo v3 to detect images.
    # Argument:
        image       : original image.
        yolo        : YOLO, yolo model.
        all_classes : all classes name.

    # Returns:
        image : processed image.
    """
    colors                 = generate_colors(all_classes)
    pimage                 = process_image(image)
    boxes, classes, scores = yolo.predict(pimage, image.shape)

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes, colors)

    return image
# -------------------------------------------------------------------------------------
if __name__ == '__main__':
    yolo        = YOLO(0.6, 0.5)
    file        = 'data/coco_classes.txt'
    all_classes = get_classes(file)
    rootdir     = 'images/test/'
    for subdir, dirs, files in os.walk(rootdir):
        for f in files:
            path  = 'images/test/' + f
            image = cv2.imread(path)
            image = detect_image(image, yolo, all_classes)
            cv2.imwrite('images/res/' + f, image)
            img   = cv2.imread('images/res/' + f)
