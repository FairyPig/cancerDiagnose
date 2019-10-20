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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class TumorClassifier(nn.Module):
    def __init__(self, reRange  = (0, 1)):
        self.reRange = reRange
        super(TumorClassifier, self).__init__()
        self.featureExtractor = lib.resnet.resnet34(pretrained=True, mid_out=True)

        self.downSample2 = nn.Sequential(#28*28
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )

        self.classifier2_feature = nn.Sequential(#28*28
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((14, 14, 14)),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((7, 7, 7)),
            nn.Conv3d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.classifier2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 1, 1),
            nn.Sigmoid()
        )

        self.downSample3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU()
        )

        self.classifier3_feature = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((7, 7, 7)),
            nn.Conv3d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.classifier3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(128, 1, 1),
            nn.Sigmoid()
        )

        self.downSample4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU()
        )

        self.classifier4_feature = nn.Sequential(
            nn.AdaptiveAvgPool3d((7, 7, 7)),
            nn.Conv3d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.classifier4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(128, 1, 1),
            nn.Sigmoid()
        )
        self.regressior = nn.Sequential(
            utils.LambdaModel(func=lambda x: x-0.5),
            nn.Linear(3, 1, bias=False), nn.Sigmoid()
        )
        self.smooth = utils.get_gaussian_kernel(3, 2, channels=1)
        # for p in self.featureExtractor.parameters():
        #     p.requires_grad=False
        # self.temp = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(40*40*1,1),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        # return [self.temp(input[:,:,:1].view((input.shape[0],-1)))],None
        '''
        0 torch.Size([8, 64, 112, 112])
        1 torch.Size([8, 64, 56, 56])
        2 torch.Size([8, 128, 28, 28])
        3 torch.Size([8, 256, 14, 14])
        4 torch.Size([8, 512, 7, 7])
        '''
        # input = input[:, 10:-5, :, :]
        # print(input.size())
        input = input.type(torch.cuda.FloatTensor)
        input = torch.clamp(input, 0, 1)
        input = utils.rerange(input, self.reRange[0], self.reRange[1])
        feats3 = [] #d nchw
        feats2 = []
        feats4 = []
        for i in range(input.shape[1]):
            in_ = input[:, i: i+1]
            in_ = in_.repeat(1, 3, 1, 1)
            in_ = utils.resnet_preprocess_batch(in_)
            r, f = self.featureExtractor(in_)

            feats2.append(torch.unsqueeze(self.downSample2(f[2].detach()), dim=0))
            feats3.append(torch.unsqueeze(self.downSample3(f[3].detach()), dim=0))
            feats4.append(torch.unsqueeze(self.downSample4(f[4].detach()), dim=0))

        feat3d2 = torch.cat(feats2, dim = 0) # d n c h w
        feat3d2 = torch.transpose(feat3d2, 0, 2) # c n d h w
        feat3d2 = torch.transpose(feat3d2, 0, 1)
        feature_2 = self.classifier2_feature(feat3d2)
        r2 = self.classifier2(feature_2)
        r2 = r2.view(-1, 1)

        feat3d3 = torch.cat(feats3, dim = 0)#d n c h w
        feat3d3 = torch.transpose(feat3d3, 0, 2)# c n d h w
        feat3d3 = torch.transpose(feat3d3, 0, 1)
        feature_3 = self.classifier3_feature(feat3d3)
        r3 = self.classifier3(feature_3)
        r3 = r3.view(-1, 1)

        feat3d4 = torch.cat(feats4, dim = 0)#d n c h w
        feat3d4 = torch.transpose(feat3d4, 0, 2)# c n d h w
        feat3d4 = torch.transpose(feat3d4, 0, 1)
        feature_4 = self.classifier4_feature(feat3d4)
        r4 = self.classifier3(feature_4)
        r4 = r4.view(-1, 1)

        r = torch.cat([r2, r3, r4], dim = 1)
        rfinal = self.regressior(r)

        r = torch.mean(r, dim=1, keepdim=True)
        return [rfinal], [r2, r3, r4, rfinal], [feature_2, feature_3, feature_4]

def initSequenceModel(seq):
    for m in seq.children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            continue
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU) or isinstance(m, nn.Sigmoid):
            continue
        if isinstance(m, nn.Sequential) or isinstance(m, nn.Module):
            initSequenceModel(m)
            continue
        assert False, type(m)

def crossEntropyLoss(y,label):
    smooth_para = 0.00
    lossT = -1*(label)* ((1-smooth_para)*torch.log(y+1e-4)+smooth_para*torch.log(1-y+1e-4)) #*frate
    lossF = -1* (1-label)*((1-smooth_para)*torch.log(1-y+1e-4)+smooth_para*torch.log(y+1e-4))#*trate
    batch_size = y.shape[0]
    loss = torch.sum(lossT+lossF)
    loss = loss/batch_size
    return loss

if __name__ == "__main__":
    model = TumorClassifier(reRange=(0.3, 0.7)).cuda()
    # initSequenceModel(model)
    train_dataset = tumorDataset.TumorDtaset("/home/afan/tumor/data/train_refine_v5", "train.csv", aug=True,
                                             mix=1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8,
        num_workers=28, drop_last=False, shuffle=True)
    frate, trate = train_dataset.distribution
    print("data distribution frate %f trate %f"%(frate, trate))
    optimizer = torch.optim.Adam([
        {"params": model.parameters()}
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    nepoach = 30
    losss = []
    print("datasetSize:", len(train_dataset))

    for i_epoach in range(nepoach):
        for i_batch, (img, label, name)in enumerate(train_loader):

            img = img.cuda()
            label = label.cuda()
            _, output, features = model(img)

            loss = [crossEntropyLoss(x, label) for x in output]

            nploss = [x.cpu().data.numpy() for x in loss]
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
        pth = './model_mask/net_%d.pth'%(i_epoach)
        print('save to', pth)
        torch.save(model.state_dict(), pth)
        benchmark.eval3(model, "/home/afan/tumor/data/train_refine_v5", "val.csv")
        scheduler.step()
        # if (i_epoach + 1) % 15 == 0:
        #     print('*' * 50)
        #     print("Test Epoch %d" % (i_epoach))
        #     benchmark.test(model, "/home/afan/tumor/data/test_refine_v5", "./output/output_"+str(i_epoach)+".csv", dataTransform = None, aug = True)
    benchmark.test_dataset(model, "/home/afan/tumor/data/test_refine_v5", "./output/output_end_3.csv", dataTransform=None,
                     aug=False)