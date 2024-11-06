import os
import sys
import matplotlib.pyplot as plt
import re
import numpy as np

def parselogfile(logfile):
    with open(logfile) as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    iters = []
    train_loss = []
    valid_loss = []
    for line in lines:
        if len(line) == 0:
            continue
        parts = re.split(" ", line)
        if "iters:" in parts:
            for i in range(len(parts)):
                if parts[i]=="iters:":
                    iters.append(int(parts[i+1]))
                if parts[i]=="train_loss:":
                    train_loss.append(float(parts[i+1]))
                if parts[i]=="valid_loss:":
                    valid_loss.append(float(parts[i+1]))
        else:
            continue
    return iters, train_loss, valid_loss

def main(argv):

    fns = argv[1:]
    fig, ax = plt.subplots(2, 1)

    for fn in fns:
        iters, train, valid = parselogfile(fn)

        s = os.path.split(os.path.dirname(fn))[-1]
        ax[0].plot(iters, train, label='{}'.format(s))
        ax[1].plot(iters, valid, label='{}'.format(s))

    plt.xlabel('iters')
    ax[0].set_ylabel('train loss')
    ax[1].set_ylabel('valid loss')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
