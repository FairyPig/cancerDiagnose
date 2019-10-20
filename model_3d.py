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
import lib.resnet3d


class TumorClassifier(nn.Module):
    def __init__(self):
        super(TumorClassifier,self).__init__()
        self.featureExtractor = nn.Sequential(
            nn.Conv3d(1,8,3,stride=1,padding=1),#40
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3,stride=1,padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3,stride=1,padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),#20
            nn.Conv3d(16,32,3,stride=1,padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3,stride=1,padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),#10
            nn.Conv3d(32,64,3,stride=1,padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),#5
            nn.Conv3d(64, 128, 3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv3d(128,64,1),
            nn.ReLU(),
            nn.Conv3d(64,1,1),
            nn.Sigmoid(),
            nn.AdaptiveMaxPool3d(1)
            #nn.AdaptiveMaxPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        # self.temp = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(40*40*1,1),
        #     nn.Sigmoid()
        # )


    def forward(self, input):
        # return [self.temp(input[:,:,:1].view((input.shape[0],-1)))],None

        #feats = self.featureExtractor(input)
        # for i,f in enumerate(feats):
        #     print(i,f.shape)
        # quit()

        '''
        0 torch.Size([1, 16, 20, 112, 112])
        1 torch.Size([1, 32, 10, 56, 56])
        2 torch.Size([1, 64, 10, 56, 28])
        3 torch.Size([1, 64, 10, 56, 14])
        4 torch.Size([1, 64, 10, 56, 7])

        '''
        #r = self.classifier(feats.view((input.shape[0],-1)))

        r = self.featureExtractor(input)
        r = r.view(-1,1)

        return [r],None


def crossEntropyLoss(y,label):
    lossT = -1*label* torch.log(y+1e-4) #*frate
    lossF = -1* (1-label)*torch.log(1-y+1e-4)#*trate
    batch_size = y.shape[0]
    loss = torch.sum(lossT+lossF)

    loss = loss/batch_size

    return loss


if __name__ == "__main__":



    model = TumorClassifier().cuda()

    train_dataset = tumorDataset.TumorDtaset("../data/train_seg","train.csv",aug=True)
    #train_dataset.csvData = train_dataset.csvData[:100]
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
        num_workers=28, drop_last=False,shuffle=True)
    frate,trate = train_dataset.distribution
    print("data distribution frate %f trate %f"%(frate,trate))
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10,0.3)
    nepoach=50
    losss = []
    print("datasetSize:",len(train_dataset))
    #benchmark.eval(model, "../data/train_img", "val.csv",dataTransform= utils.toCubeData)
    for i_epoach in range(nepoach):
        for i_batch,(img,label)in enumerate(train_loader):
            img = utils.toCubeData(img)
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
        benchmark.eval(model,"../data/train_seg","val.csv",dataTransform= utils.toCubeData)
        scheduler.step()

