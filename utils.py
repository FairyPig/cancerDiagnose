import numpy as np
import torchvision.transforms as transforms
import torch
import random
import cv2
import os
import csv
import pydicom
import math
from scipy.misc import imresize
from PIL import Image, ImageEnhance

def restnet_preprocess(array,aug = False):
    # im = Image.open(r"./data/PIOD/target/2008_000002.jpg")
    # in_ = np.array(im, dtype=np.float32)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    # r = normalize(transforms.ToTensor()(im))
    # print(r.type())
    array = np.array(array,dtype=np.float32)
    #im = Image.open(r"./data\PIOD\target/2008_000002.jpg")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    array /= 255
    if aug:
        array += np.random.normal(loc=0,scale=0.05,size=array.shape)
    array = np.transpose(array,(2,0,1))#c*h*w
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i in range(3):
        array[i]-=means[i]
        array[i] /= stds[i]
    return array

def resnet_preprocess_batch(array):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    for i in range(array.shape[0]):
        for j in range(3):
            array[i][j] -= means[j]
            array[i][j] /= stds[j]
    return array

def squeezeChannel(d):
    assert d.shape[2] == 1 and d.shape[3] == 1
    d = torch.squeeze(d, 1)
    d = torch.squeeze(d,1)
    return d

def crossEntropyLoss(y,label):
    lossT = -1*label* torch.log(y+1e-4)
    lossF = -1* (1-label)*torch.log(1-y+1e-4)
    batch_size = y.shape[0]
    loss = torch.sum(lossT+lossF)

    loss = loss/batch_size

    return loss

def scaleData(matrix,targetChannel,targetSize):
    matrix = np.transpose(matrix, (2, 0, 1))#wch
    m_ = []
    for i in range(matrix.shape[0]):#scale w c
        m_.append(cv2.resize(matrix[i], (targetSize, targetChannel), interpolation=cv2.INTER_LINEAR))
    matrix = np.array(m_)
    matrix = np.transpose(matrix, (1, 2, 0))#chw

    m_=[]
    for i in range(targetChannel):#schle wh
        m_.append(cv2.resize(matrix[i], (targetSize, targetSize), interpolation=cv2.INTER_LINEAR))
    matrix = np.array(m_)

    return matrix

def randomCrop(data, channelShift = 5,imgShift=40):
    srcShape = data.shape
    data = np.pad(data,((0,0),(imgShift//2,imgShift//2),(imgShift//2,imgShift//2)),mode="constant",constant_values=0)
    channelshift_ = random.randint(1,channelShift)
    if random.random()>0.5:
        data = scaleData(data[channelshift_:],srcShape[0],srcShape[1])
    else:
        data = scaleData(data[:-channelshift_],srcShape[0],srcShape[1])
    if imgShift <=1:return data
    shifty = random.randint(1,imgShift)
    if random.random() > 0.5:
        data = scaleData(data[:,shifty:], srcShape[0], srcShape[1])
    else:
        data = scaleData(data[:,:-shifty], srcShape[0], srcShape[1])

    shiftx = random.randint(1,imgShift)
    if random.random() > 0.5:
        data = scaleData(data[:,:,shiftx:], srcShape[0], srcShape[1])
    else:
        data = scaleData(data[:,:,:-shiftx], srcShape[0], srcShape[1])
    return data

def randRange(low,high):
    r = random.random()
    r *= (high - low)
    r += low

def center_crop(image, crop_size):
    h, w = image.shape
    top = int((h - crop_size) / 2)
    left = int((w - crop_size) / 2)
    bottom = top + crop_size
    right = left + crop_size
    image = image[top:bottom, left:right]
    return image

def scale_augmentation(image, scale_size):
    crop_size = 224
    image = imresize(image, (scale_size, scale_size))
    image = center_crop(image, crop_size)
    return image

def scale_random(data, scale_size1, scale_size2):
    data_new = []
    scale_size = np.random.randint(scale_size1, scale_size2)
    if scale_size >= 224:
        for i in range(data.shape[0]):
            img = Image.fromarray(np.uint8(data[i] * 255))
            img = scale_augmentation(img, scale_size)
            img = np.asarray(img) / 255.0
            data_new.append(img)
    else:
        for i in range(data.shape[0]):
            img_zero = np.zeros(data[0].shape)
            img = Image.fromarray(np.uint8(data[i] * 255))
            img = imresize(img, (scale_size, scale_size))
            img = np.asarray(img) / 255.0
            left = int((224 - scale_size)/2)
            img_zero[left:left+scale_size,left:left+scale_size] = img
            data_new.append(img_zero)
    return np.array(data_new)

def HuShift(data,shiftRange=(-0.1,0.1),alphaRange = (0.7,1.3)):
    shift = randRange(*shiftRange)
    alpha = randRange(*alphaRange)
    data = data + torch.FloatTensor(shift)
    data = torch.clamp(data,0,1)
    data = torch.pow(data,alpha)
    return data

def randomLight(data):
    data_new = []
    random_factor_1 = np.random.randint(0, 10) / 10.  # 随机因子
    random_factor_2 = np.random.randint(10, 13) / 10.  # 随机因子
    random_factor_3 = np.random.randint(10, 13) / 10.  # 随机因子
    random_factor_4 = np.random.randint(0, 5) / 10.  # 随机因子
    for i in range(data.shape[0]):
        img = Image.fromarray(np.uint8(data[i] * 255))
        color_image = ImageEnhance.Color(img).enhance(random_factor_1)  # 调整图像的饱和度
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor_2)  # 调整图像的亮度
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor_3)  # 调整图像对比度
        img_new = ImageEnhance.Sharpness(contrast_image).enhance(random_factor_4) # 调整图像锐度
        img_new = np.asarray(img_new) / 255.0
        data_new.append(img_new)
    return np.array(data_new)

class LambdaModel(torch.nn.Module):
    def __init__(self, func):
        super(LambdaModel, self).__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)


def rerange(data,low,high):
    assert torch.max(data) <= 1 and torch.min(data) >= 0
    data -= low
    data[data < 0] = 0
    data /= (high-low)
    data[data > 1] = 1

    assert torch.max(data) <= 1 and torch.min(data) >= 0
    return data
def refineData(dataroot,outputroot):
    filenames = os.listdir(dataroot)
    if os.path.exists(outputroot) is False:
        os.mkdir(outputroot)
    for filename in filenames:
        #filename="1E5193A6-B8C9-4F3D-A2C9-EBB0398C649D.npy"
        filePath = os.path.join(dataroot,filename)
        data = np.load(filePath).item()['data']
        data = np.array(data)
        #data = rotateData(data,-90)
        #select target
        #data = rotateData(data,20)

        deltas = []
        for i in range(data.shape[0]-1):
            delta = np.abs(data[i+1]-data[i])
            delta = np.clip(delta,0,0.1)
            deltas.append(delta)
        delta_map = np.sum(deltas,axis=0,keepdims=True)
        #print(delta_map.shape)
        delta_map[delta_map<0.1*1]=0
        delta_map[delta_map>0]=1
        # delta_map = data.var(axis=0,keepdims=True)
        # delta_map /= delta_map.max()
        delta_map = delta_map[0]
        delta_map = cv2.dilate(delta_map,np.ones((5,5)))
        #print(delta_map.shape,delta_map.dtype)

        image, contours, hierarchy = cv2.findContours((delta_map*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        spaces = []
        for c in contours:
            t = np.sum(cv2.drawContours(np.zeros_like(delta_map), [c], -1, 1, -1))
            spaces.append(t)
        body_id = np.argmax(spaces)
        mask =cv2.drawContours(np.zeros_like(delta_map), [contours[body_id]], -1, 1, -1)
        delta_map = delta_map*mask
        # cv2.imwrite("./output/test.png",(delta_map*255).astype(np.uint8))
        # quit()
        delta_map = np.expand_dims(delta_map,0)
        data = data*delta_map


        #algin gt
        #data = rotateData(data,20)
        rect = cv2.minAreaRect(contours[body_id])
        angle = rect[-1]
        if (0 < abs(angle) and abs(angle) <= 45):

            angle = angle;
        elif (45 < abs(angle) and abs(angle) < 90):
            angle = 90 - abs(angle)
        data = rotateData(data,angle)
        #print(rect)
        x,y,w,h = cv2.boundingRect((mask*255).astype(np.uint8))
        #cv2.imwrite("./output/test.jpg",(mask*255)[y:y+h,x:x+w])
        srcShape = data.shape
        data = data[:,y:y+h,x:x+w]

        if h/w>1.1:
            data = rotateData(data,90)
            print("rotate")

        data = scaleData(data,data.shape[0],srcShape[1])

        # cv2.imwrite("./output/1.jpg",(data[5]*255).astype(np.uint8))
        # quit()

        np.save(os.path.join(outputroot,filename),{"data":data})

def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4

    M = cv2.getRotationMatrix2D(center, angle, scale) #5

    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7

def rotateData(data,angle):
    assert len(data.shape) == 3
    m = []
    for i in range(data.shape[0]):
        m.append(rotate(data[i],angle))
    return np.array(m)

def getTemplateData(data):
    data = data.cpu().data.numpy()
    assert len(data.shape) == 4
    r = []
    for sample in data:
        sample_clip = np.clip(sample,0,0.5)
        maxIndex = np.argmax(np.sum(sample_clip,axis=(1,2)))
        #print(maxIndex)
        temp = sample[maxIndex]
        temp = [temp,temp,temp]
        temp = np.transpose(temp,(1,2,0))
        temp = restnet_preprocess(temp*255)
        r.append(temp)
    return torch.from_numpy(np.array(r)).cuda().float()


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def toCubeData(data):
    data = [scaleData(d[:,:,:].cpu().data.numpy(),40,40) for d in data]
    data = torch.unsqueeze(torch.from_numpy(np.array(data)).cuda().float(),dim=1)
    return data

def readCsv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        csvData = list(csv.reader(f))
        csvData = csvData[1:]
        return csvData

def writeCsv(path,csvData):
    f= open(path,'w')

    f.write("id,ret\r\n")
    for k,v in csvData:
        f.write("%s,%s\r\n"%(k,str(v)))
    f.close()

def loadDcms(dir):
    filenames = os.listdir(dir)
    filenames.sort()
    imgs = []
    for filename in filenames:
        # filename = "../data/train\\0B055F4C-6F1D-42E1-815C-7AD4E45DB9AB/33038abe-47b2-4887-b23e-7ad4811599fb_00031.dcm"
        # print(filename)
        dcmfile = pydicom.read_file(os.path.join(dir, filename))  #
        dcmImg = np.array(dcmfile.pixel_array, dtype=np.float32)
        assert dcmImg.shape == (512, 512)
        # cv2.imwrite("./output/test.jpg",(dcmImg*255).astype(np.uint8))
        # quit()
        # dcmImg = dcmImg[128:128 + 256, 30:30 + 256]
        # dcmImg = (dcmImg * 255).astype(np.uint8)
        # dcmImg = cv2.resize(dcmImg, (224, 224), interpolation=cv2.INTER_LINEAR)
        imgs.append(dcmImg)
    return np.array(imgs)-1024


def maxSegMask(img):
    assert len(img.shape) == 2
    spaces = []
    image, contours, hierarchy = cv2.findContours((img * 255).astype(np.uint8), cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        t = np.sum(cv2.drawContours(np.zeros_like(img), [c], -1, 1, -1))
        spaces.append(t)
    if np.sum(spaces) == 0:
        return np.zeros_like(img)
    body_id = np.argmax(spaces)
    #cv2.imshow("b",cv2.drawContours(np.zeros_like(average), contours, -1, 1, -1))
    mask = cv2.drawContours(np.zeros_like(img), [contours[body_id]], -1, 1, -1)
    return mask

def sortImgs(imgs):
    deltas = []
    for i in range(len(imgs)):
        delta = np.sum(np.abs(imgs[i] - imgs[i - 1]))
        deltas.append(delta)
    cut_index = np.argmax(deltas)
    if cut_index != 0:
        imgs = np.concatenate([imgs[cut_index:] , imgs[:cut_index]],axis=0)
    return np.array(imgs)


if __name__ == "__main__":
    #refineData("../data/train_img","../data/train_img_refine")
    refineData("../data/test_img", "../data/test_img_refine")




