from modelfeast import *
import torch
import numpy as np
import torch.nn as nn
import lib.resnet
import torch.nn.functional
import tumorDataset
import benchmark
from skimage import transform
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# class TumorClassifier(nn.Module):
#     def __init__(self, reRange=(0, 1)):
#         self.reRange = reRange
#         self.base_resnet = resnet18_3d(n_classes=400, in_channels=32)
#         self.classifier = nn.Sequential(
#             nn.Linear(400, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         input = input.type(torch.cuda.FloatTensor)
#         resnet_out = self.base_resnet(input)
#         predict = self.classifier(resnet_out)
#         return predict

def crossEntropyLoss(y,label):
    smooth_para = 0.00
    lossT = -1*(label)* ((1-smooth_para)*torch.log(y+1e-4)+smooth_para*torch.log(1-y+1e-4)) #*frate
    lossF = -1* (1-label)*((1-smooth_para)*torch.log(1-y+1e-4)+smooth_para*torch.log(y+1e-4))#*trate
    batch_size = y.shape[0]
    loss = torch.sum(lossT+lossF)
    loss = loss/batch_size
    return loss

if __name__ == "__main__":
    # model = TumorClassifier(reRange=(0.3, 0.7)).cuda()
    model = resnet18_3d(n_classes=2, in_channels=1).cuda()
    train_dataset = tumorDataset.TumorDtaset("/home/afan/tumor/data/train_refine_v5", "train.csv", aug=False,
                                             mix=1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32,
        num_workers=28, drop_last=False, shuffle=True)

    frate, trate = train_dataset.distribution
    print("data distribution frate %f trate %f"%(frate, trate))
    optimizer = torch.optim.Adam([
        {"params": model.parameters()}
    ], lr=1e-5)
    nepoach = 30
    losss = []
    print("datasetSize:", len(train_dataset))

    for i_epoach in range(nepoach):
        for i_batch, (img, label, name)in enumerate(train_loader):
            img = img.cuda()
            img = img.type(torch.cuda.FloatTensor)
            label = label.cuda()
            output = model(img)

            loss = [crossEntropyLoss(x, label) for x in output]
            nploss = [x.cpu().data.numpy()for x in loss]

            losss.append(nploss)
            if len(losss) == 10:
                print(i_epoach, i_batch, np.average(losss))
                losss.clear()

            loss_total = None
            for i, l in enumerate(loss):
                if i == 0:
                    loss_total = l
                else:
                    loss_total = loss_total + l
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()
        pth = './model_resnet/net_%d.pth'%(i_epoach)
        print('save to', pth)
        torch.save(model.state_dict(), pth)
        benchmark.eval(model, "/home/afan/tumor/data/train_refine_v5", "val.csv")