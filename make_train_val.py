import numpy as np
import utils
import random
if __name__ == "__main__":
    src = "./train_label.csv"
    csvdata = utils.readCsv(src)
    random.shuffle(csvdata)

    val = csvdata[:100]
    train = csvdata[100:]
    utils.writeCsv("./train.csv",train)
    utils.writeCsv("./val.csv",val)