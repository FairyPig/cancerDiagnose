import torch
import numpy as np
import cv2
import os
import torch.nn as nn
import torchvision
import lib.resnet
import torch.nn.functional
import tumorDataset
import benchmark
import utils


class AdaptivePooling(nn.Module):
    def __init__(self):
        super(AdaptivePooling,self).__init__()
    def forward(self, input):
        if self.training:
            return nn.functional.adaptive_avg_pool2d(input,3)
        else:
            return input

class TumorClassifier(nn.Module):
    def __init__(self):
        super(TumorClassifier,self).__init__()
        self.featureExtractor = lib.resnet.resnet18(pretrained=False,in_feature=40,mid_out=True)
        self.classifier = nn.Sequential(
            #nn.AdaptiveAvgPool2d(3),
            AdaptivePooling(),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(512,32,1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        _,feats=self.featureExtractor(input)
        r = self.classifier(feats[-1])#nchw
        r = torch.squeeze(r,1)
        r = torch.squeeze(r,1)
        #r = torch.squeeze(r,1)

        return [r],_


def crossEntropyLoss(y,label):
    lossT = -1*label* torch.log(y+1e-4) #*frate
    lossF = -1* (1-label)*torch.log(1-y+1e-4)#*trate
    batch_size = y.shape[0]
    loss = torch.sum(lossT+lossF)

    loss = loss/batch_size

    return loss


if __name__ == "__main__":



    model = TumorClassifier().cuda().train()

    train_dataset = tumorDataset.TumorDtaset("../data/train_seg","train.csv",aug=True)
    # d = train_dataset[2][0]
    # for i in range(40):
    #     y = d[i]
    #     y[y<0]=0
    #     y[y>1]=1
    #     cv2.imwrite("./output/%2d.png"%(i),(y*255).astype(np.uint8))
    #
    # quit()

    #train_dataset.csvData = train_dataset.csvData[:100]
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8,
        num_workers=8, drop_last=False,shuffle=True)
    frate,trate = train_dataset.distribution
    print("data distribution frate %f trate %f"%(frate,trate))
    optimizer = torch.optim.SGD(model.parameters(),lr=3e-4,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10,0.3)
    nepoach=30
    losss = []
    print("datasetSize:",len(train_dataset))
    benchmark.eval(model, "../data/train_seg", "val.csv")
    for i_epoach in range(nepoach):
        for i_batch,(img,label)in enumerate(train_loader):
            #continue
            img = img.cuda()
            label = label.cuda()
            output,_ = model(img)

            loss = [crossEntropyLoss(x,label) for x in output]
            nploss = [x.cpu().data.numpy()for x in loss]

            losss.append(nploss)
            if len(losss)==10:
                print(i_epoach, i_batch, np.average(losss))
                losss.clear()
            # if i_epoach == 0 and i_batch < 100:
            #     loss = loss*0.1
            loss_total =None
            for i,l in enumerate(loss):
                if i == 0:
                    loss_total = l
                else:
                    loss_total = loss_total+l
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()
        pth = './model/net_%d.pth'%(i_epoach)
        print('save to',pth)
        torch.save(model.state_dict(),pth)
        benchmark.eval(model,"../data/train_seg","val.csv")
        scheduler.step()

