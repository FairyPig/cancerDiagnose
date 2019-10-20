import utils
import cv2
import os
import numpy as np

import scipy.ndimage
import matplotlib.pyplot as plt
import time
from skimage import measure, morphology
import threading
import multiprocessing
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import joblib
import shutil
def plot_3d(image, threshold=-300,id="1"):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    #p = p[:,::-1,:]
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    fig = plt.gcf()
    plt.show()
    fig.savefig('./output/%s.png'%(id), dpi=100)

def keepBody(data,low,high):
    deltas = []

    average = np.max(data,axis=0)+np.average(data,axis=0)*0.6

    average[average>-130]=1
    average[average<-130]=0
    #
    average = cv2.erode(average,kernel=np.ones((5,5)))
    average = cv2.dilate(average, np.ones((7, 7)))
    #average = cv2.erode(average,kernel=np.ones((5,5)),iterations=3)

    # print(delta_map.shape,delta_map.dtype)

    mask = utils.maxSegMask(average)
    #cv2.imshow("c",mask)
    #cv2.waitKey(0)
    mask = np.expand_dims(mask,axis=0)
    data = data *mask
    data[np.ones(data.shape)*mask == 0]=-1023


    # maskinner = np.min(data,axis=0)
    # gt = maskinner>-0
    # lt = maskinner<-0
    # maskinner[gt]=0
    # maskinner[lt] = 1
    # maskinner = maskinner*cv2.erode(mask[0],kernel=np.ones((5,5)),iterations=10)
    # cv2.imshow("a",cv2.erode(mask[0],kernel=np.ones((5,5)),iterations=5))
    # cv2.waitKey(0)
    mask_enrode = cv2.erode(mask[0],kernel=np.ones((5,5)),iterations=3)
    data = data*np.array([mask_enrode])
    data[np.ones_like(data)*np.array([mask_enrode]) == 0]=-1023

    mask_bone = np.average(data,axis=0)
    mask_bone[mask_bone<100]=0
    mask_bone[mask_bone>0]=1
    #mask_bone = np.array(mask_bone*255,dtype=np.uint8)
    mask_bone = cv2.erode(mask_bone,kernel=np.ones((5,5)))
    mask_bone = cv2.dilate(mask_bone, kernel=np.ones((5,5)))
    #cv2.imshow("bone",mask_bone)


    do_skip = False
    segs = []
    for i,layer in enumerate(data*(1-np.array([mask_bone]))):
        outer_bone = cv2.threshold(layer,140,1,type=cv2.THRESH_BINARY)[1]
        #cv2.imshow("outer bone",outer_bone)
        l = np.clip(layer,low,high)-(low)
        l /= (high-low)
        l*=(1-outer_bone)
        #cv2.imshow("a",l)
        l0 = l.copy()
        l[l>0]=1

        l = cv2.dilate(l,kernel=np.ones((3,3)),iterations=1)
        l = cv2.erode(l,kernel=np.ones((3,3)),iterations=7)
        if np.sum(l)< 100:
            l = np.zeros_like(l)
        #cv2.imshow("b", l)
        l = cv2.dilate(l,kernel=np.ones((3,3)),iterations=4)
        #cv2.imshow("c", cv2.threshold(l*l0,1e-10,1,type=cv2.THRESH_BINARY)[1])
        max_seg = utils.maxSegMask(l)
        segs.append(max_seg)
        if do_skip or i>len(data)/3 and len(np.where(segs[-1]+segs[-2] ==2)[0]) == 0:
            l = np.zeros_like(l)
            do_skip=True
        #cv2.imshow("d",  l0*utils.maxSegMask(l))

        #cv2.waitKey(0)
        data[i] *= l*utils.maxSegMask(l)

    # for i,layer in enumerate(data):
    #     mask = np.array(layer)
    #     gt = mask>-1023
    #     lt = mask <= -1023
    #     mask[gt]=1
    #     mask[lt]=0
    #     image, contours, hierarchy = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE,
    #                                                   cv2.CHAIN_APPROX_SIMPLE)
    #
    #     mask = cv2.drawContours(np.zeros_like(mask),contours, -1, 1, -1)
    #     mask = cv2.dilate(mask,kernel=np.ones((5,5)))
    #     mask = cv2.erode(mask, kernel=np.ones((3, 3)), iterations=3)
    #     data[i][mask == 0]=-1024
    #     # cv2.imshow("a",data[i]/data[i].max()*2)
    #     # cv2.imshow("b",mask)
    #     # cv2.waitKey(0)

    return np.array(data)



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
    imgs = utils.scaleData(imgs[8:-5],40,224)
    imgs = keepBody(imgs,low,high)
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
    #cv2.setNumThreads(56)
    # dir  =r"templateData\0AADE9F9-BAB0-4712-AAEB-16FFDC61C4A8"
    # imgs = utils.loadDcms(dir)
    # imgs = np.array(imgs)
    # imgs = utils.sortImgs(imgs)
    # imgs = utils.scaleData(imgs,40,224)
    # imgs = utils.scaleData(imgs[8:-5],40,224)
    # imgs = keepBody(imgs)
    # # imgs[imgs>100]=0
    # #imgs[imgs>100]=0
    # plot_3d(imgs.astype(np.int32), 1)

    data_root = "../data/train/"
    targetRoot = "../data/train_seg"
    data_root = "../data/test/"
    targetRoot = "../data/test_seg"
    ids = os.listdir(data_root)
    threads = []
    #assert "7AC108D7-A69B-418D-96BC-599208ECA9D8" in ids
    #print(len(ids))
    if os.path.exists(targetRoot):
        shutil.rmtree(targetRoot)
        time.sleep(1)
    os.mkdir(targetRoot)
    joblib.Parallel(n_jobs=56)(joblib.delayed(process)(
        os.path.join(data_root, id), os.path.join(targetRoot, id + ".npy"))
        for id in ids
    )

    # for id in ids:
    #     # if id == "17CDA3B8-DD5D-4F8B-83EB-D6B8A50B489A":
    #     #     print("aaa")
    #     # else:
    #     #     continue
    #     #id = '7BDCA098-C6EE-42FD-8B91-FCE696D72465'
    #     #data = process(os.path.join(data_root,id))
    #     #process(os.path.join(data_root,id),os.path.join(targetRoot,id+".npy"))
    #     t = multiprocessing.Process(target=process,
    #                          args=(os.path.join(data_root,id),os.path.join(targetRoot,id+".npy")))
    #     threads.append(t)
    #     if (len(threads) == 32):
    #         #threads.clear()
    #         for t in threads:
    #             t.start()
    #         for t in threads:
    #             t.join()
    #         threads.clear()
    # for t in threads:
    #     t.start()
    # for t in threads:
    #     t.join()
    print("finish")