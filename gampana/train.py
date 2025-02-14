import torch
import torch.nn as nn

import os
from tqdm import tqdm

from gampixpy import config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from dataloader import DataLoader
from network import EnergyRegressionModel

def main(args):
    if args.readout_config == "":
        readout_config = config.default_readout_params
    else:
        readout_config = config.ReadoutConfig(args.readout_config)

    dl = DataLoader(args.train, readout_config)
    model = EnergyRegressionModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-5, weight_decay=5e-4)

    for batch in dl.iter_batches():
        
        # print (image)
        continue
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type = str,
                        default = '../test_inputs/neutron_0-5GeV/',
                        help = "input train data (hdf5)")
    parser.add_argument('--test', type = str,
                        default = '../test_inputs/',
                        help = "input test data (hdf5)")

    parser.add_argument('-n', '--nEpochs', type = int,
                        default = 5,
                        help = "maximum number of epochs to train")
    parser.add_argument('-c', '--checkpoint', type = str,
                        default = '.',
                        help = "checkpoint to load")
    parser.add_argument('-f', '--checkpoint_period', type = int,
                        default = 1,
                        help = "save a checkpoint every N epochs")
    parser.add_argument('-o', '--output', type = str,
                        default = '.',
                        help = "output prefix directory")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    parser.add_argument('-r', '--readout_config',
                        type = str,
                        default = "",
                        help = 'readout configuration yaml')
    
    args = parser.parse_args()

    main(args)
