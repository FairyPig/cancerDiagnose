import numpy as np
import csv
import os
def showLabel(labelPath):
    with open(labelPath, 'r', encoding='utf-8-sig') as f:
        csvData = list(csv.reader(f))
        csvData = csvData[1:]
    count_t = 0
    count_f = 0
    for d in csvData:
        if int(d[1]) == 0:
            count_f +=1
        elif int(d[1]) == 1:
            count_t +=1
    assert count_t + count_f == len(csvData)
    print(labelPath)
    print("data count", len(csvData), "positive:", count_t, "negative:", count_f)

def showDeepth(dataroot):
    ids = os.listdir(dataroot)
    ds =[]
    for id in ids:
        ds.append(len(os.listdir(os.path.join(dataroot,id))))
    print(dataroot)
    print(np.average(ds))

if __name__ == "__main__":
    showLabel("train_label.csv")
    showLabel("train1.csv")
    showLabel("train2.csv")
    showLabel("train.csv")

    showLabel("val.csv")
    showLabel("output.csv")
    showLabel("output1.csv")
    showLabel("output2.csv")
    showLabel("output_cut.csv")

    #showDeepth("../data/train")
    #showDeepth("../data/test")
    print("npys:",len(os.listdir("../data/train_seg")))
