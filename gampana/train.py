import numpy as np

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

    dl = DataLoader(args.train, readout_config, batch_size = 4)
    model = EnergyRegressionModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1.e-4,
                                 # weight_decay=5e-4,
                                 )

    import time
    loss_series = torch.empty(0,).to(device)

    t0 = time.time()
    last_time = t0
    n_iter = 0
    for batch in dl.iter_batches():
        this_time = time.time() - t0
        # print (this_time, this_time - last_time)
        last_time = this_time

        x, y = batch
        y = y[:,None]
        
        # print ("input", x.features)
        out = model(x)
        pred = 1000*out.features

        # print (out)

        frac_err = torch.mean(torch.abs((pred - y)/y))

        loss = torch.mean(torch.pow(pred - y, 2))

        loss_series = torch.cat((loss_series, loss[None].detach()))

        print ("iter", n_iter, "loss", round(loss.item(), 2), round(frac_err.item(), 2))

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        
        n_iter += 1
        if args.max_iter and n_iter > args.max_iter:
            break

    np.save(args.output, loss_series.detach().cpu().numpy())
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type = str,
                        # default = '/sdf/data/neutrino/dougl215/gampixpy/neutron_0-5GeV_batch2_250212/gampixD/',
                        default = '/lscratch/dougl215/slurm_job_id_64303752/',
                        help = "input train data (hdf5)")
    parser.add_argument('--test', type = str,
                        default = '../test_inputs/',
                        help = "input test data (hdf5)")

    parser.add_argument('-n', '--nEpochs', type = int,
                        default = 5,
                        help = "maximum number of epochs to train")
    parser.add_argument('-m', '--max_iter', type = int,
                        default = None,
                        help = "maximum number of epochs to train")
    parser.add_argument('-c', '--checkpoint', type = str,
                        default = '.',
                        help = "checkpoint to load")
    parser.add_argument('-f', '--checkpoint_period', type = int,
                        default = 1,
                        help = "save a checkpoint every N epochs")
    parser.add_argument('-o', '--output', type = str,
                        default = '.',
                        help = "training log file")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    parser.add_argument('-r', '--readout_config',
                        type = str,
                        default = "",
                        help = 'readout configuration yaml')
    
    args = parser.parse_args()

    main(args)
