import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def binarization(img):  # reduce channel to 1 and binarize each grey scaled pixel.
    ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    return thresh

def ada_binarization(img):
    return cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)

def noise_remove(img, h):  # h is the parameter that determines how strong the filter is.
    dst = cv.fastNlMeansDenoising(img,None,h,21,7)
    return dst

def thinning(img):
    d_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    e_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    dilation = cv.dilate(img, d_kernel, iterations=1)
    erosion = cv.erode(dilation, e_kernel, iterations=1)
    return erosion

def skel(img):
    skeleton = np.zeros(img.shape,np.uint8)
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img = cv.bitwise_not(img)
    while True:
        open = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        temp = cv.subtract(img, open)
        eroded = cv.erode(img, element)
        skeleton = cv.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv.countNonZero(img) == 0:
            break
    skeleton = cv.bitwise_not(skeleton)
    return skeleton

def center_and_resize(img,threshold,new_size,character_size):

    def center_of_mass(img):  # find the center of mass(black dots) of the image
        h,w = img.shape[0],img.shape[1]
        x = 0
        y = 0
        total = 0

        for i in range(h):
            for j in range(w):
                temp = img[j,i]
                if temp > threshold:
                    continue
                total += 1
                x += j
                y += i

        center_x = x//total
        center_y = y//total

        if total <= 0:
            center_x = int(w * 0.5)
            center_y = int(h * 0.5)

        return center_x,center_y

    def bounding_box(img, threshold):  # find the boundaries of the black area
        h,w = img.shape[0],img.shape[1]
        x_min = w-1
        x_max = 0
        y_min = h-1
        y_max = 0

        for i in range(h):
            for j in range(w):
                temp = img[j,i]
                if temp > threshold:
                    continue

                x_min = min(j,x_min)
                x_max = max(j,x_max)
                y_min = min(i,y_min)
                y_max = max(i,y_max)

        return x_min,x_max,y_min,y_max

    def cropping_box(img,rect,pt):  # find the smallest square centered at the center of mass
        h,w = img.shape[0],img.shape[1]
        size = max(h,w)
        min_r = size
        for r in range(size):
            x1 = pt[0] - r
            x2 = pt[0] + r
            y1 = pt[1] - r
            y2 = pt[1] + r

            xc1 = rect[0]
            xc2 = rect[1] + 1
            yc1 = rect[2]
            yc2 = rect[3] + 1

            if x1 > xc1: continue
            if y1 > yc1: continue
            if x2 < xc2: continue
            if y2 < yc2: continue

            min_r = r
            break

        new_x_min = pt[0] - min_r
        new_y_min = pt[1] - min_r
        new_x_max = pt[0] + min_r
        new_y_max = pt[1] + min_r
        return img[new_y_min:new_y_max,new_x_min:new_x_max]

    def downscale(img,character_size):
        return cv.resize(img,(character_size,character_size),interpolation=cv.INTER_AREA)

    def pad(img,shape):
        h,w = img.shape[0],img.shape[1]
        left = (shape-w)//2
        top = (shape-h)//2
        right = left if not (shape - w) & 2 else left + 1
        bottom = top if not (shape - h) & 2 else top + 1
        return cv.copyMakeBorder(img,top,bottom,left,right,cv.BORDER_CONSTANT,None,255)

    pt = center_of_mass(img)
    boundaries = bounding_box(img,threshold)
    cropped = cropping_box(img,boundaries,pt)
    scaled = downscale(cropped,character_size)
    padded = pad(scaled,new_size)
    return padded


def preprocess_procedure_1(img,threshold,size,ch_size):  # binarization -> noise remove -> thinning -> cropping
    img = np.array(img) if not isinstance(img,type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = binarization(img)
    noised = noise_remove(thresh,10)
    thinned = thinning(noised)
    processed = center_and_resize(thinned,threshold,size,ch_size)
    return processed


def preprocess_procedure_2(img,threshold,size,ch_size):  # binarization -> noise remove -> skeletonization -> cropping
    img = np.array(img) if not isinstance(img, type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = binarization(img)
    noised = noise_remove(thresh, 10)
    thinned = skel(noised)
    processed = center_and_resize(thinned,threshold,size,ch_size)
    return processed


def preprocess_procedure_3(img,threshold,size,ch_size):  # cropping -> binarization -> noise remove -> thinning
    img = np.array(img) if not isinstance(img, type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rescaled = center_and_resize(img,threshold,size,ch_size)
    thresh = binarization(rescaled)
    noised = noise_remove(thresh, 10)
    processed = thinning(noised)
    return processed


def preprocess_procedure_4(img,threshold,size,ch_size):  # cropping -> binarization -> noise remove -> skeletonization
    img = np.array(img) if not isinstance(img, type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rescaled = center_and_resize(img,threshold,size,ch_size)
    thresh = binarization(rescaled)
    noised = noise_remove(thresh, 10)
    processed = skel(noised)
    return processed

def vanilla(img,threshold,size,ch_size):
    img = np.array(img) if not isinstance(img, type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = center_and_resize(img,threshold,size,ch_size)
    return img


    
if __name__ == "__main__":
    img = cv.imread("test4.png")
    proc1 = preprocess_procedure_1(img,177,25,25)
    print(proc1.shape)
    proc2 = preprocess_procedure_2(img,177,28,20)
    proc3 = preprocess_procedure_3(img,177,28,20)
    proc4 = preprocess_procedure_4(img,177,28,20)
    original = vanilla(img,177,28,20)
    titles = ['Original Image', 'proc1', 'proc2', 'proc3', 'proc4', 'vanilla']
    images = [img, proc1, proc2, proc3, proc4, original]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
