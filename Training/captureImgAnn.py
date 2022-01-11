import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from keras.models import load_model


# Check if the two digits are part of the same number
def getIntersection(Ia, Ib):
    if Ib[0] > Ia[1] or Ia[0] > Ib[1]:
        return -1
    else:
        Os = max(Ia[0], Ib[0])
        Oe = min(Ia[1], Ib[1])
        return (Os, Oe)


# Crops every digit in an image so that it would be easy for classification model
def cropImage(timg, tol=0):
    # Convert to binary
    # (thresh, timg) = cv2.threshold(timg, 127, 255, cv2.THRESH_BINARY)
    #  Crop only ones from binary
    mask = timg > tol
    timg = timg[np.ix_(mask.any(1), mask.any(0))]
    # Add padding to the image
    timg = np.pad(timg, pad_width=round(timg.shape[0] * 0.1), mode='constant', constant_values=0)
    timg = timg.astype(np.uint8)
    # Resize the image to 200*200 pixels
    timg = cv2.resize(timg, (28, 28), interpolation=cv2.INTER_AREA)
    timg[timg > 0] = 255
    # timg = cv2.blur(timg,(25,25))
    return timg


# Process the image in such a way that only those number shoed up which are neaded to be calculated.

def processImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rimg, gimg, bimg = cv2.split(img)
    hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsvimg)

    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # grayImg[grayImg<150] = 255
    # grayImg[grayImg>150] = 0
    # return grayImg

    u_red = np.array([255, 100, 100])
    l_red = np.array([100, 0, 0])
    mask = cv2.inRange(img, l_red, u_red)
    blur = cv2.medianBlur(mask, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (13, 13))
    dilation = cv2.dilate(blur, kernel, 1)

    # rmimg = morphology.remove_small_objects(dilation.astype(bool),min_size=250,connectivity=2).astype(np.uint8)*255
    # arr = np.nonzero(rmimg)
    # l_range = arr[1].min()
    # h_range = arr[1].max()

    arr = np.nonzero(dilation)
    unique, counts = np.unique(arr[1], return_counts=True)
    if len(counts) > 0:
        center = unique[np.argmax(counts == counts.max())]
        l_range = center - 50
        h_range = center + 50

        blur[:, :] = 0
        blur[:, l_range:h_range] = 255
        s[s > 50] = 255
        outimg = cv2.bitwise_and(blur, s)
        outimg[outimg < 127] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        outimg = cv2.dilate(outimg, kernel, 1)
    else:
        outimg = np.zeros_like(blur)

    return outimg


def processImg(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([155,25,0])
    upper_red = np.array([179,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    # maskt = np.dstack([maskt, maskt, maskt]) / 255
    # out = img * maskt

    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 150

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    img2 = img2.astype(np.uint8)
    return img2



if __name__ == "__main__":

    # Load image from directory
    img = cv2.imread('data_test/1.jpg')

    pimg = processImg(img)

    # Separted every digit in an image
    _, thresh = cv2.threshold(pimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connectivity = 8  # You need to choose 4 or 8 for connectivity type
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    e_labels = []
    for i in range(1, len(centroids)):
        oneLabel = np.zeros_like(labels)
        oneLabel[labels == i] = 255
        oneLabel = cropImage(oneLabel)
        oneLabel = oneLabel.reshape((28, 28, 1))
        e_labels.append(oneLabel)

    # This is the prediction code
    classifier = load_model("ann_model.h5")
    imgs = np.array(e_labels)
    if len(imgs) > 0:
        timgs = imgs.reshape(len(imgs),784)
        preds = np.argmax(classifier.predict(timgs), axis=-1)
        preds = np.insert(preds, 0, -1, axis=0)

        e_data = {}
        for i in range(1, len(centroids)):
            obj = lambda: None
            obj.center = centroids[i]
            obj.pred = preds[i]
            obj.stats = stats[i]
            obj.range = sorted([stats[i][1], stats[i][1] + stats[i][3]])
            obj.column = stats[i][0]
            e_data[i] = obj

        numbers = []
        check_arr = [True] * len(e_data.values())
        for indi, i in enumerate(e_data.values()):
            temp = []
            temp.append(i)
            if check_arr[indi]:
                for indj, j in enumerate(e_data.values()):
                    if i != j:
                        res = getIntersection(i.range, j.range)
                        if res != -1:
                            temp.append(j)
                            check_arr[indj] = False
                check_arr[indi] = False
                numbers.append(temp)

        # This is combination of digits, code.
        c_numbers = []
        for i in numbers:
            size = len(i)
            temp = i
            temp.sort(key=lambda x: x.column)
            num = int(''.join([str(i[k].pred) for k in range(size)]))
            c_numbers.append(num)
        result = sum(c_numbers)

        cv2.putText(img, f"Result: {result}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA,
                    False)
        # plt.imshow(img)
        for obj in e_data.values():
            x, y, w, h = (obj.stats[0], obj.stats[1], obj.stats[0] + obj.stats[2], obj.stats[1] + obj.stats[3])
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(img, f"{obj.pred}", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        cv2.LINE_AA, False)

        # Display the resulting frame

    resizedImg = cv2.resize(img,(540,700))
    cv2.imshow('ANN result', resizedImg)

    cv2.waitKey(0)