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
import random


class TumorClassifier(nn.Module):
    def __init__(self):
        super(TumorClassifier,self).__init__()
        self.featureExtractor = lib.resnet.resnet34(pretrained=False,in_feature=40,mid_out=True)

        self.classifier2 = nn.Sequential(
            nn.Conv2d(128,64,3,stride=1,padding=3,dilation=3),
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(64, 1, 7),
            nn.Sigmoid()
        )


        self.classifier3 = nn.Sequential(
            nn.Conv2d(256,16,1),
            #nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(16, 1, 14),
            nn.Sigmoid()

        )
        self.classifier4 = nn.Sequential(
            nn.Conv2d(512, 1, 7),
            nn.Sigmoid()

        )
        self.classifier_final = nn.Sequential(
            nn.Conv2d(3,1,1),
            nn.Sigmoid()
        )



    def forward(self, input):
        _,feats=self.featureExtractor(input)
        # for i,f in enumerate(feats):
        #     print(i,f.shape)
        # quit()

        '''
        0 torch.Size([1, 64, 112, 112])
        1 torch.Size([1, 64, 56, 56])
        2 torch.Size([1, 128, 28, 28])
        3 torch.Size([1, 256, 14, 14])
        4 torch.Size([1, 512, 7, 7])

        '''
        r2_ = self.classifier2(feats[2])
        r3_ = self.classifier3(feats[3])
        r4_ = self.classifier4(feats[4])

        #r2,r3,r4 = [torch.nn.functional.adaptive_max_pool2d(x,1) for x in [r2_,r3_,r4_]]

        rfinal_ = self.classifier_final(torch.cat([r2_,r3_,r4_],dim=1))

        # r2, r3, r4,rfinal = [torch.nn.functional.adaptive_max_pool2d(x, 1) for x in [r2_, r3_, r4_,rfinal_]]
        # r = [utils.squeezeChannel(d) for d in [r2,r3,r4,rfinal]]
        r = [utils.squeezeChannel(x) for x in [r2_,r3_,r4_]]
        return r,None#[r2_,r3_,r4_,rfinal]


def crossEntropyLoss(y,label):
    lossT = -1*label* torch.log(y+1e-4) *frate
    lossF = -1* (1-label)*torch.log(1-y+1e-4)*trate
    batch_size = y.shape[0]
    loss = torch.sum(lossT+lossF)

    loss = loss/batch_size

    return loss


if __name__ == "__main__":



    model = TumorClassifier().cuda()

    train_dataset = tumorDataset.TumorDtaset("../data/train_img_refine","train.csv",aug=True)
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
        num_workers=10, drop_last=False,shuffle=True)
    frate,trate = train_dataset.distribution
    print("data distribution frate %f trate %f"%(frate,trate))
    optimizer = torch.optim.SGD(model.parameters(),lr=3e-5,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10,0.3)
    nepoach=30
    losss = []
    print("datasetSize:",len(train_dataset))
    benchmark.eval(model, "../data/train_img_refine", "val.csv")
    for i_epoach in range(nepoach):
        for i_batch,(img,label)in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()
            output,outmap = model(img)

            loss = [crossEntropyLoss(x,label) for x in output]
            nploss = [x.cpu().data.numpy()for x in loss]
            #outmap = [x.cpu().data.numpy() for x in outmap]
            # for i in range(len(outmap[0])):
            #     m = [cv2.resize(outmap[j][i][0],(224,224)) for j in range(len(outmap))]
            #     cv2.imwrite("./output/%5d.jpg"%(random.randrange(0,1000)),(np.concatenate(m,axis=1)*255).astype(np.uint8))
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
        benchmark.eval(model,"../data/train_img_refine","val.csv")
        scheduler.step()

