import pydicom
import os
import shutil
import time
import cv2
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import utils
import joblib
import threading
# def dcmToImg():
#     '''
#
#     胸部:常规胸部CT检查分别用纵隔窗及肺窗观察,纵隔窗可观察心脏、大血管的位置,纵隔内淋巴结的大小,纵隔内肿块及这些结构的比邻关系,设定纵隔窗可用窗宽300 Hu~500 Hu,窗位30 Hu~50 Hu,
#     肺部：窗宽1300 Hu~1 700 Hu,窗位-600 Hu~-800 Hu,在上述基本窗宽的基础上,若观察肺裂和肺血管,可调窄窗宽和调低窗位,对肿块形态，分叶，胸膜凹陷征，毛刺征增的观察肺窗比纵隔窗更为清晰,
#     腹部检查：常设定窗宽为300 Hu~500 Hu,窗位30 Hu~50 Hu,
#     肝脾CT检查应适当变窄窗宽以便更好发现病灶，窗宽为100 Hu~200 Hu,窗位为30 Hu~45 Hu,
#
#     '''
#
#     windowLen = 250
#     windowCenter = 30
#     pids = os.listdir(srcRoot)
#     if os.path.exists(targetRoot):
#         shutil.rmtree(targetRoot)
#         time.sleep(1)
#     os.mkdir(targetRoot)
#
#     for p in pids:
#         srcDir = os.path.join(srcRoot, p)
#         targetDir = os.path.join(targetRoot, p)
#         os.mkdir(targetDir)
#         filenames = os.listdir(srcDir)
#         for filename in filenames:
#             print(filename)
#             dcmfile = pydicom.read_file(os.path.join(srcDir, filename))
#             dcmImg = np.array(dcmfile.pixel_array, dtype=np.float32)
#             dcmImg = (dcmImg - 1024 + windowCenter) / windowLen
#             dcmImg[dcmImg > 1] = 1
#             dcmImg[dcmImg < 0] = 0
#             assert dcmImg.shape == (512, 512)
#             dcmImg = dcmImg[128:128 + 256, 30:30 + 256]
#             dcmImg = (dcmImg * 255).astype(np.uint8)
#             dcmImg = cv2.cvtColor(dcmImg, cv2.COLOR_GRAY2BGR)
#             cv2.imwrite(os.path.join(targetDir, filename + ".png"), dcmImg)



def convertData_old(srcRoot,targetRoot):

    # srcRoot = "../data/train"
    # targetRoot = "../data/train_img"

    windowLen = 120
    windowMin = 0
    pids = os.listdir(srcRoot)
    if os.path.exists(targetRoot):
        shutil.rmtree(targetRoot)
        time.sleep(1)
    os.mkdir(targetRoot)
    targetSize=224
    targetChannel = 40
    for p in pids:
        srcDir = os.path.join(srcRoot, p)
        # targetDir = os.path.join(targetRoot, p)
        # os.mkdir(targetDir)
        filenames = os.listdir(srcDir)

        imgs = []
        filenames.sort()

        for filename in filenames:
            #filename = "../data/train\\0B055F4C-6F1D-42E1-815C-7AD4E45DB9AB/33038abe-47b2-4887-b23e-7ad4811599fb_00031.dcm"
            #print(filename)
            dcmfile = pydicom.read_file(os.path.join(srcDir, filename))#
            dcmImg = np.array(dcmfile.pixel_array, dtype=np.float32)
            dcmImg = (dcmImg - 1024 - windowMin) / windowLen
            dcmImg[dcmImg > 1] = 1
            dcmImg[dcmImg < 0] = 0
            assert dcmImg.shape == (512, 512)
            # cv2.imwrite("./output/test.jpg",(dcmImg*255).astype(np.uint8))
            # quit()
            #dcmImg = dcmImg[128:128 + 256, 30:30 + 256]
            #dcmImg = (dcmImg * 255).astype(np.uint8)
            dcmImg = cv2.resize(dcmImg,(224,224),interpolation=cv2.INTER_LINEAR)
            imgs.append(dcmImg)


        deltas = []
        for i in range(len(imgs)):
            delta = np.abs(imgs[i]-imgs[i-1])
            deltas.append(delta)
        cut_index = np.argmax(deltas)
        if cut_index != 0:
            imgs = imgs[cut_index:]+imgs[:cut_index]
        matrix = np.array(imgs)
        assert np.max(matrix)<=1 and np.min(matrix)>=0
        # matrix = np.transpose(matrix,(2,0,1))
        # m_=[]
        # for i in range(targetSize):
        #     m_.append(cv2.resize(matrix[i],(targetSize,targetChannel),interpolation=cv2.INTER_LINEAR))
        # matrix = np.array(m_)
        #
        #
        # matrix = np.transpose(matrix,(1,2,0))
        matrix = utils.scaleData(matrix,40,224)
        #matrix = scipy.ndimage.zoom(matrix,(40/matrix.shape[0],1,1),mode='nearest')
        assert np.max(matrix)<=1 and np.min(matrix)>=0

        assert matrix.shape == (targetChannel,targetSize,targetSize),matrix.shape
        targetPath = os.path.join(targetRoot, p+".npy")
        print(targetPath)
        np.save(targetPath,{"data":matrix})
        # interpolater = scipy.interpolate.RegularGridInterpolator((
        #     np.linspace(0,32),np.linspace(0,224),np.linspace(0,224)),
        #
        # )



def process(dir,targetDir=None):
    if os.path.exists(targetDir):
        return
    print("process",dir)
    low = 20
    high = 70
    id = os.path.split(dir)[-1]
    imgs = utils.loadDcms(dir)
    imgs = np.array(imgs)
    imgs = utils.sortImgs(imgs)
    imgs = utils.scaleData(imgs,40,224)
    #imgs = utils.scaleData(imgs[8:-5],40,224)
    #imgs = keepBody(imgs,low,high)
    #print("f")
    #plot_3d(imgs.astype(np.int32), 1,id)
    imgs = imgs-low
    imgs /= (high-low)
    imgs =np.array(imgs)
    if targetDir != None:
        temp = os.path.split(targetDir)[0]

        # if not os.path.exists(temp):
        #     os.mkdir(temp)
        #     time.sleep(1)
    np.save(targetDir,{"data":imgs})

    #return imgs



if __name__ == "__main__":
    # src = "templateData/00CAD9C8-D90B-43C4-9964-79CAE6493DA7/28ff388d-a8d3-4999-95cf-5d071300892b_00011.dcm"
    # file = pydicom.read_file(src)
    # img = np.array(file.pixel_array,dtype=np.float32)
    # print(img.shape)
    # print(type(img))
    # print(img.max(),img.min())
    # cv2.imshow("a",img/np.max(img))
    # img_ = (img-1024+30)/200
    # print(img_[10][10],img[10][10])
    # img_[img_<0]=0
    # img_[img_>1]=1
    # cv2.imshow("b",(img_*255).astype(np.uint8))
    # cv2.waitKey(0)
    # quit()
    srcRoot = "../data/train"
    targetRoot = "../data/train_img"
    srcRoot = "../data/test"
    targetRoot = "../data/test_img"
    #convertData_old(srcRoot,targetRoot)

    if not os.path.exists(targetRoot):
        os.mkdir(targetRoot)

    ids = os.listdir(srcRoot)

    if os.path.exists(targetRoot):
        shutil.rmtree(targetRoot)
        time.sleep(1)
    os.mkdir(targetRoot)
    joblib.Parallel(n_jobs=28)(joblib.delayed(process)(
        os.path.join(srcRoot, id), os.path.join(targetRoot, id + ".npy"))
        for id in ids
    )



    # srcRoot = "../data/test"
    # targetRoot = "../data/test_img"
    # convertData(srcRoot,targetRoot)