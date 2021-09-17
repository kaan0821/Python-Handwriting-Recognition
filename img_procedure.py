import numpy
from PIL import Image
from preprocess import *
import cv2 as cv

def getImg(img_dir):
    try:
        # Get grayscale image matrix from directory:
        img = Image.open(img_dir).convert('L')
    except:
        return None
    # If image size not 25x25, resize:
    if img.size != (25, 25):
        img = img.resize((25,25))
    #img = cv.imread(img_dir)
    #这两行可以删掉，照常运行
    #这里输出的是25x25的numpy array
    #img = preprocess_procedure_1(img,177,25,25)
    
    # Return image matrix list:
    img = numpy.array(img).reshape(1,-1).tolist()
    img_list = []
    for line in img:
        for value in line:
            img_list.append(value)
    return img

def getImgDir(argumans):
    # We get the image directory:
    if len(argumans) < 2:
        return None
    return argumans[1]
