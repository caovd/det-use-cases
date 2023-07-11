import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def plot_original_logs():
    data = pd.read_csv('logs.txt', sep=" ", header=None)
    print('Orignal logs: ', data)

    val_loss = []
    train_loss = []
    for i in range(len(data.loc[:,4])):
        if data.loc[i,2] == "validation":
            val_loss.append(data.loc[i,4])
        if data.loc[i,2] == "training":
            train_loss.append(data.loc[i,4])
            
    # Convert str to float
    val_loss = [float(x) for x in val_loss]
    train_loss = [float(x) for x in train_loss]
    
    return val_loss, train_loss

def plot_det_logs():
    return

def plot_both_logs():
    return

if __name__ == '__main__':
    parser = ArgumentParser('Print plot arguments') 
    parser.add_argument('-s', '--source', default='original', help='Log source - options: original, det, both')
    args = parser.parse_args().__dict__

    if args['source'] == 'original':
        val_loss, train_loss = plot_original_logs()
        plt.plot(val_loss, 'o-r')
        plt.plot(train_loss, 'o-b')
        plt.xlabel('Epoch')
        plt.ylabel('Original loss')
        plt.legend(["val", "train"], loc ="upper right")
        plt.show()
    elif args['source'] == 'det':
        plot_det_logs()
    else:
        plot_both_logs()

    