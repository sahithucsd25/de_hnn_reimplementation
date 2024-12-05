#!/usr/bin/env python

import sys
sys.path.insert(0, 'src')

from train import train
from utils import plot_losses

sys.path.remove('src')

def main(targets):
    test = True
    n_epochs = 100

    if len(targets) == 1:
        if targets[0].isdigit(): 
            n_epochs = int(targets[0])
        elif targets[0].lower() == 'true':
            test = True
        else:
            test = False
    elif len(targets) == 2:
        n_epochs = int(targets[1])
        if targets[0].lower() == 'true':
            test = True
        else:
            test = False

    train(test=test, n_epochs=n_epochs)
    
    if not test:
        plot_losses('net')
        plot_losses('node')
    
    return

if __name__ == '__main__':
    # run via:
    # python main.py test n_epochs
    targets = sys.argv[1:]
    main(targets)