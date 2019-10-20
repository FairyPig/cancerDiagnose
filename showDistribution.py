import numpy as np
import cv2
import os
import csv
import shutil
import matplotlib.pyplot as plt
import random

def show_avg(dataRoot,labelPath,outputdir,isTest=False):
    print("process ",labelPath)
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)
    with open(labelPath, 'r', encoding='utf-8-sig') as f:
        csvData = list(csv.reader(f))
        csvData = csvData[1:]

    avgImg = [np.zeros((40,224,224)) for j in range(2)]
    for i,line in enumerate(csvData):
        if i % 10 == 0:
            print(i,i/len(csvData))
        data = np.load(os.path.join(dataRoot, line[0] + ".npy")).item()
        data = data['data']
        data = np.clip(data,0,1)
        label = int(line[1])
        avgImg[label] = (avgImg[label]*i + data)/(i+1)

    if isTest:
        for j in range(40):
            cv2.imwrite(os.path.join(outputdir, "avg_%d_%d.png" % (j, 0)), (avgImg[0][j]+avgImg[1][j])/2 * 255)
    else:
        for i in range(2):
            for j in range(40):
                cv2.imwrite(os.path.join(outputdir,"avg_%d_%d.png"%(j,i)),avgImg[i][j]*255)



def do_avg():
    for i in range(40):
        im1 = (cv2.imread("./temp1/avg_%d_%d.png"%(i,0))).astype(np.float32)
        im2 = cv2.imread("./temp1/avg_%d_%d.png" % (i, 1)).astype(np.float32)
        cv2.imwrite("./temp1/avg_%d_%d.png"%(i, 2),(im1+im2)/2)

def show_gradient(srcDir,targetDir):
    if os.path.exists(targetDir):
        shutil.rmtree(targetDir)
    os.makedirs(targetDir)

    filenames = os.listdir(srcDir)

    for f in filenames:
        img = cv2.imread(os.path.join(srcDir,f))
        #img = np.average(img,axis=2)
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # 对x求一阶导
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        g = np.sqrt(grad_x*grad_x+grad_y*grad_y)
        g[g<60]=0
        cv2.imwrite(os.path.join(targetDir,f),g)

def matchGradient(dataRoot,labelPath,gradientPath,targetRoot):
    with open(labelPath, 'r', encoding='utf-8-sig') as f:
        csvData = list(csv.reader(f))
        csvData = csvData[1:]

    hints = []
    hintfiles = os.listdir(gradientPath)
    hintfiles.sort()
    hintfiles = ["avg_%d_0.png"%(i) for i in range(40)]
    print(hintfiles)
    hint_mask = cv2.imread("./temp/hint_mask2.png").astype(np.float32)
    hint_mask[hint_mask<128]=0
    hint_mask[hint_mask>0]=1
    for hf in hintfiles:
        himg = cv2.imread(os.path.join(gradientPath,hf)).astype(np.float32)
        himg *= hint_mask
        #himg[himg>0]=1
        himg = cv2.dilate(himg,np.ones((3,3)))
        hints.append(himg)

    result = {}
    random.shuffle(csvData)
    for i,line in enumerate(csvData):
        if i % 10 == 0:
            print(i,i/len(csvData))
        data = np.load(os.path.join(dataRoot, line[0] + ".npy")).item()
        data = data['data']
        data = np.clip(data,0,1)
        gsum = 0
        for i in range(20,40):
            img = data[i]
            img = (img *255).astype(np.uint8)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # 对x求一阶导
            grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            g = np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float64)
            g[g<60]=0
            g[g>60]=1
            g = cv2.dilate(g,np.ones((3,3)))
            cor = np.sum(
                hints[i]*(hints[i] * g)
                #hints[i]*g
            )
            gsum += cor
        result[line[0]] = gsum/40
    print(result)
    plt.hist(result.values(),30)
    plt.show()
    np.save(os.path.join(targetRoot,"cor.npy"),{"cor":result})

def showHist(dataPath):
    cors = np.load(os.path.join(dataPath)).item()['cor']
    cors_dict = cors.copy()
    cors = np.array(list(cors.values()))/10000


    plt.hist(cors,100)
    plt.xlim(np.min(cors),np.max(cors))
    plt.show()
    print(np.sum(cors<70)/cors.size)

    outputfile = "./output1.csv"
    f= open(outputfile,'w')

    f.write("id,ret\r\n")
    tCount = 0
    for k,v in cors_dict.items():
        t = 1 if v/10000 > 5500 else 0
        if t == 1:tCount +=1
        f.write("%s,%d\r\n"%(k,t))
    f.close()

if __name__ == "__main__":
    # show_avg("../data/train_img","train1.csv","temp/train1")
    # show_avg("../data/train_img", "train2.csv", "temp/train2")
    #show_avg("../data/test_img", "output.csv", "temp/test1",isTest=True)
    # show_gradient("temp/train1","temp/gradient1")
    # show_gradient("temp/train2","temp/gradient2")
    # show_gradient("temp/test1","temp/gradient_test")
    #matchGradient("../data/test_img", "output.csv","temp/gradient_test","./temp")
    showHist("G:\\tumor\\output_0.76.csv")
    # do_avg()