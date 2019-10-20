import numpy as np
import csv


if __name__ == "__main__":
    # csv = np.loadtxt("train_label.csv",delimiter=",",usecols=(0,1),encoding='utf-8-sig',skiprows=1)
    # print(csv)
    with open("train_label.csv",'r',encoding='utf-8-sig') as f:
        csv_data = list(csv.reader(f))
        print(csv_data)