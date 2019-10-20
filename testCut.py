import os
import csv
import numpy as np
import matplotlib.pyplot as plt
with open("G:\\tumor\\output_end_4.csvs", 'r', encoding='utf-8-sig') as f:
    csvData = list(csv.reader(f))
    csvData = csvData[1:]
csvData.sort(key=lambda x:float(x[1]))
print(len(csvData))

#print(csvData)
f = open("./output_cut.csv",'w')
f.write("id,ret\r")
negativeRate = 0.57
print(csvData[int(len(csvData)*negativeRate)])
for i,l in enumerate(csvData):
    label = 0 if i/len(csvData) < negativeRate else 1
    f.write("%s,%d\r" % (l[0], label))
f.close()

p = [float(x[1]) for x in csvData]
plt.hist(p,100)
gcf = plt.gcf()
gcf.savefig('./output/test.png', dpi=100)
plt.show()