import cv2
import os
import numpy as np
import shutil
import utils

if __name__ == "__main__":
    datadir = "../data/train_img/"
    outputdir = "./output/train/"
    # datadir = "../data/test_img/"
    # outputdir = "./output/test/"
    ids = os.listdir(datadir)

    for j,id in enumerate(ids):
        if j > 50:quit()
        srcPath = os.path.join(datadir,id)
        targetdir = os.path.join(outputdir,os.path.splitext(id)[0])
        if os.path.exists(targetdir):
            shutil.rmtree(targetdir)
        os.makedirs(targetdir)
        data = np.load(srcPath).item()
        data = data['data']
        data = np.clip(data,0,1)
        #print(data.max(),data.min())
        for i in range(len(data)):
            cv2.imwrite(os.path.join(targetdir,"%2d.jpg"%(i)), utils.rotate((data[i] * 255).astype(np.uint8),0))

# d = train_dataset[2][0]
# for i in range(40):
#     y = d[i]
#     y[y<0]=0
#     y[y>1]=1
#     cv2.imwrite("./output/%2d.png"%(i),(y*255).astype(np.uint8))
#
# quit()