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
        self.classifier = nn.Sequential(
            nn.Conv2d(40,1,224),
            nn.Sigmoid()
        )

    def forward(self, input):
        # return [self.temp(input[:,:,:1].view((input.shape[0],-1)))],None
        input = torch.clamp(input,0,1)

        r = self.classifier(input)
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
    # model.load_state_dict(torch.load('./model/net_%d.pth'%(0)))
    # d = model.classifier.children()
    # for i,c in enumerate(d):
    #     if i == 0:
    #         print(c.weight)
    #         w = c.weight.cpu().data.numpy()
    #         print(w.shape)
    #         w = w[0]
    #         t = []
    #         for j in range(40):
    #             temp = w[j].copy()
    #             temp -= np.min(w[i])
    #             temp /= temp.max()
    #             t.append(temp)
    #             cv2.imwrite("./temp/%d.png"%(j),(temp*255).astype(np.uint8))
    #         cv2.imwrite("./temp/a.png",np.average(t,axis=0)*255)
    #
    # quit()


    train_dataset = tumorDataset.TumorDtaset("../data/test_img",None,aug=False,
                                             simiRoot="../data/test_img",similabel="./outputr3.csvs",simithres=0.1,

                                             )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8,
        num_workers=28, drop_last=False,shuffle=True)
    frate,trate = train_dataset.distribution
    print("data distribution frate %f trate %f"%(frate,trate))
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10,0.3)
    nepoach=50
    losss = []
    print("datasetSize:",len(train_dataset))
    #benchmark.eval(model, "../data/train_img", "val.csv",dataTransform= utils.toCubeData)
    for i_epoach in range(nepoach):
        for i_batch,(img,label)in enumerate(train_loader):
            #img = utils.toCubeData(img)
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
        benchmark.eval(model,"../data/train_img","val.csv")
        scheduler.step()

