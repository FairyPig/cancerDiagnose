import torch
import csv
import numpy as np
import os
import utils
import tumorDataset
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def eval(model,dataroot,labelPath,dataTransform = None):
    model.eval()
    with open(labelPath, 'r', encoding='utf-8-sig') as f:
        csvData = list(csv.reader(f))
        csvData = csvData[1:]

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    samples = len(csvData)
    losss = []
    for line in csvData:

        data = np.load(os.path.join(dataroot, line[0] + ".npy"))
        label = int(line[1])

        tensordata = torch.from_numpy(np.expand_dims(data,0).astype(np.float32)).cuda()
        if dataTransform is not None:
            tensordata = dataTransform(tensordata)
        r, _ = model(tensordata)
        labrlTensor = torch.from_numpy(np.array([[label]])).cuda().float()
        loss = [utils.crossEntropyLoss(y,labrlTensor) for y in r]
        loss = [x.cpu().data.numpy() for x in loss]
        losss.append(np.average(loss))
        r = [x.cpu().data.numpy() for x in r]
        print([np.squeeze(x).tolist() for x in r],label)
        r = r[-1]
        assert r.shape == (1,1),r.shape
        r = np.sum(r)
        r= 1 if r>0.5 else 0
        #print(label,r)
        if label == 0:
            if r == 0:
                tn +=1
            else:
                fp +=1
        elif label == 1:
            if r == 0:
                fn +=1
            else:
                tp +=1
        else:
            assert False
    print(tp,tn,fp,fn)
    precisiont = tp/(tp+fp+1e-10)
    recallt = tp/(tp+fn+1e-10)
    f1t = 2*precisiont*recallt/(precisiont+recallt+1e-10)
    precisionf = tn/(tn+fn+1e-10)
    recallf = tn/(tn+fp+1e-10)
    f1f =2*precisionf*recallf/(precisionf+recallf+1e-10)
    f1avg = (f1f+f1t)/2
    print("positive precision %f recall %f f1 %f "%(precisiont, recallt, f1t))
    print("negative precision %f recall %f f1 %f "%(precisionf, recallf, f1f))
    print("final score %f "%(f1avg))
    print("loss:%f"%(np.average(losss)))

def eval3(model,dataroot,labelPath,dataTransform = None):
    model.eval()
    with open(labelPath, 'r', encoding='utf-8-sig') as f:
        csvData = list(csv.reader(f))
        csvData = csvData[1:]

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    samples = len(csvData)
    losss = []
    for line in csvData:

        data = np.load(os.path.join(dataroot, line[0] + ".npy"))
        label = int(line[1])

        tensordata = torch.from_numpy(np.expand_dims(data,0).astype(np.float32)).cuda()
        if dataTransform is not None:
            tensordata = dataTransform(tensordata)
        r, _, _ = model(tensordata)
        labrlTensor = torch.from_numpy(np.array([[label]])).cuda().float()
        loss = [utils.crossEntropyLoss(y,labrlTensor) for y in r]
        loss = [x.cpu().data.numpy() for x in loss]
        losss.append(np.average(loss))
        r = [x.cpu().data.numpy() for x in r]
        print([np.squeeze(x).tolist() for x in r],label)
        r = r[-1]
        assert r.shape == (1,1),r.shape
        r = np.sum(r)
        r= 1 if r>0.5 else 0
        #print(label,r)
        if label == 0:
            if r == 0:
                tn +=1
            else:
                fp +=1
        elif label == 1:
            if r == 0:
                fn +=1
            else:
                tp +=1
        else:
            assert False
    print(tp,tn,fp,fn)
    precisiont = tp/(tp+fp+1e-10)
    recallt = tp/(tp+fn+1e-10)
    f1t = 2*precisiont*recallt/(precisiont+recallt+1e-10)
    precisionf = tn/(tn+fn+1e-10)
    recallf = tn/(tn+fp+1e-10)
    f1f =2*precisionf*recallf/(precisionf+recallf+1e-10)
    f1avg = (f1f+f1t)/2
    print("positive precision %f recall %f f1 %f "%(precisiont, recallt, f1t))
    print("negative precision %f recall %f f1 %f "%(precisionf, recallf, f1f))
    print("final score %f "%(f1avg))
    print("loss:%f"%(np.average(losss)))
def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

def test_dataset(model, dataroot, outputfile, dataTransform=None, aug=True):
    test_datasets = tumorDataset.TumorDtaset(dataroot, "./test.csv", aug=aug, mix=0)
    predicts = {}
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=8,
        num_workers=8, drop_last=False, shuffle=True)
    repeat_time = 1
    if aug:
        repeat_time = 5
    for times in range(repeat_time):
        for i_batch, (imgs, labels, name) in enumerate(test_loader):
            if i_batch % 100 == 0:
                print("Epoch:[ " + str(times) + " ] Batch: [ " + str(i_batch) + "/" + str(len(test_loader)) + " ]")
            images = make_variable(imgs, volatile=True)
            predict, _ = model(images)
            predict = predict[0].cpu().data.numpy()
            for i in range(predict.shape[0]):
                if name[i] not in predicts:
                    predicts[name[i]] = 0.0
                predicts[name[i]] += predict[i]
    f = open(outputfile, 'w')
    f.write("id,ret\r\n")
    for k, v in predicts.items():
        f.write("%s,%d\r\n" % (k, np.round(predicts[k]/float(repeat_time))))
    f.close()
    f = open(outputfile + "s", 'w')
    f.write("id,ret\r\n")
    for k, v in predicts.items():
        f.write("%s,%f\r\n" % (k, predicts[k]/float(repeat_time)))
    f.close()

def test(model, dataroot, outputfile, dataTransform=None, aug=True):
    testlist = os.listdir(dataroot)
    predicts = {}
    model.eval()
    for i, line in enumerate(testlist):
        if i % 10 == 0:
            print(" %d %f"%(i,i/len(testlist)))

        id = os.path.splitext(line)[0]
        data = np.load(os.path.join(dataroot, line))
        # data = data['data']
        # data = utils.scaleData(data[:,:,:112], 40, 224)
        rs = []
        if aug:
            for i in range(5):
                temp = data.copy()

                temp = utils.randomCrop(temp, 2, 20)
                tensordata = torch.from_numpy(np.expand_dims(temp, 0).astype(np.float32)).cuda()
                if dataTransform is not None:
                    tensordata = dataTransform(tensordata)
                r, _ = model(tensordata)
                r = [x.cpu().data.numpy() for x in r]
                r = r[-1]

                assert r.shape == (1,1),r.shape
                r = np.sum(r)
                rs.append(r)
        else:
            temp = data.copy()
            tensordata = torch.from_numpy(np.expand_dims(temp, 0).astype(np.float32)).cuda()
            if dataTransform is not None:
                tensordata = dataTransform(tensordata)
            r, _, _ = model(tensordata)
            r = [x.cpu().data.numpy() for x in r]
            r = r[-1]

            assert r.shape == (1, 1), r.shape
            r = np.sum(r)
            rs.append(r)
        r = np.average(rs)
        predicts[id] = r


    # with open("./submit_example.csv", 'r', encoding='utf-8-sig') as f:
    #     csvData = list(csv.reader(f))
    #     csvData = csvData[1:]


    f = open(outputfile,'w')

    f.write("id,ret\r\n")
    for k, v in predicts.items():
        f.write("%s,%d\r\n"%(k,round(predicts[k])))
    f.close()

    f = open(outputfile+"s",'w')

    f.write("id,ret\r\n")
    for k, v in predicts.items():
        f.write("%s,%f\r\n" % (k, predicts[k]))
    f.close()


if __name__ == "__main__":
    import model_pretrain as model_
    model = model_.TumorClassifier().cuda().eval()
    model.load_state_dict(torch.load("./model/net_29.pth"))
    #eval(model,"../data/train_img","val.csv")
    test_dataset(model, "/home/afan/tumor/data/test_refine_v5", "output/output_4.csv", dataTransform=None, aug=True)
    #=utils.getTemplateData
    #utils.toCubeData

