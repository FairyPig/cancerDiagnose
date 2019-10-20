import torch
from PIL import Image
import cv2
import torch.utils.data
import csv
import numpy as np
import os
import utils
import random
class TumorDtaset(torch.utils.data.Dataset):
    def __init__(self,dataroot,labelPath=None,aug=False,
                 simiRoot=None,similabel=None,simithres=0,
                mix= 0
                 ):
        super(TumorDtaset,self).__init__()
        self.aug = aug
        if labelPath != None:
            with open(labelPath, 'r', encoding='utf-8-sig') as f:
                csvData = list(csv.reader(f))
                #print(csv_data)
            self.csvData = csvData[1:]
            self.csvData = [[os.path.join(dataroot,l[0]),l[1]] for l in self.csvData]
        else:
            self.csvData = []
        self.mix = mix
        if simiRoot != None:
            with open(similabel, 'r', encoding='utf-8-sig') as f:
                sdata = list(csv.reader(f))
            sdata = sdata[1:]
            simiCount =0
            temp = [0,0]
            for l in sdata:
                c = float(l[1])
                if c < simithres[0] or c > 1-simithres[1]:
                    c = 0 if c<0.5 else 1
                    temp[c]+=1
                    simiCount +=1
                    for i in range(2):
                        self.csvData.append(
                            [os.path.join(simiRoot,l[0]),c]
                        )
            print("simiCount:",simiCount,temp)
        self.dataroot = dataroot

        count_t=0
        count_f = 0
        for d in self.csvData:
            l = int(d[1])
            if (l == 0):
                count_f += 1
            elif l == 1:
                count_t +=1
        assert count_t+count_f == len(self.csvData)

        self.distribution = [count_f/len(self.csvData),count_t/len(self.csvData)]
    def __len__(self):
        #if not self.aug:
        return len(self.csvData)
        #return len(self.csvData)*2

    def getItem(self,item):
        line = self.csvData[item]
        data = np.load(line[0]+".npy").astype(np.float32)
        #data = data['data']
        #data = utils.scaleData(data[:,:,:112],40,224)
        label = line[1]
        return data,np.array([label],dtype=np.float32)

    def getItemLabel(self,item):
        line = self.csvData[item]
        label = line[1]
        return int(label)

    def __getitem__(self, id):
        item = id
        if not self.mix:
            d,l = self.getItem(item)
        else:
            d1,l1 = self.getItem(id)
            otherId = random.randint(0, len(self.csvData) - 1)
            while self.getItemLabel(otherId) != np.sum(l1) and self.mix == 2:
                otherId = random.randint(0,len(self.csvData)-1)
            mixPara = random.betavariate(0.2,0.2)
            d2,l2 = self.getItem(otherId)
            d = mixPara*d1 + (1-mixPara)*d2
            l = mixPara*l1 + (1-mixPara)*l2
            #return d,l
        if self.aug:
            d = utils.rotateData(d, random.randint(-10, 10))
            d = utils.randomCrop(d,channelShift=3,imgShift=20)
        return d, np.array(l, dtype=np.float32)

    def flip_lr(self,data):
        return np.array(data[:,:,::-1])
    def flip_ud(self,data):
        return np.array(data[:,::-1,:])
    def rotate(self,data):
        return np.transpose(data,(0,2,1))

class TumorDtaset_triple(TumorDtaset):
    def __init__(self,*a,**args):
        super(TumorDtaset_triple,self).__init__(*a,**args)

    def __getitem__(self, item):
        thisLabel = self.getItemLabel(item)
        Tlabel = -1
        Tid = -1
        while Tlabel != thisLabel:
            Tid = random.randint(0,len(self.csvData)-1)
            Tlabel = self.getItemLabel(thisLabel)

        FLable = thisLabel
        Fid = -1

        while FLable == thisLabel:
            Fid = random.randint(0,len(self.csvData)-1)
            FLable = self.getItemLabel(Fid)

        data_anchor = self.do_crop(self.getItem(item))
        data_t = self.do_crop(self.getItem(Tid))
        data_f = self.do_crop(self.getItem(Fid))
        return data_anchor + data_t + data_f

    def do_crop(self,d):
        if not self.aug:
            return d
        d[0] = utils.rotateData(d[0], random.randint(-10, 10))
        d[0] = utils.randomCrop(d[0],channelShift=3,imgShift=20)
        return d