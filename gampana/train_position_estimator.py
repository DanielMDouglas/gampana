import numpy as np

import torch
import torch.nn as nn

import os
from tqdm import tqdm

from gampixpy import config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from dataloader import PointSourceLoader
from network import PositionEstimatorModel

def main(args):
    if args.readout_config == "":
        readout_config = config.default_readout_params
    else:
        readout_config = config.ReadoutConfig(args.readout_config)

    dl = PointSourceLoader(args.train, readout_config, batch_size = 128)
    model = PositionEstimatorModel().to(device)
    if args.checkpoint:
        # load from checkpoint instead of initializing from scratch
        print (args.checkpoint)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay,
                                 # weight_decay=5e-4,
                                 )

    import time
    loss_series = torch.empty(0,).to(device)
    loss_log_file = os.path.join(args.output_directory,
                                 "loss.npy")

    t0 = time.time()
    last_time = t0
    # iter_per_epoch = 1000

    for n_epoch in range(args.nEpochs):
        dl.set_file_load_order()
        n_iter = 0
        pbar = tqdm(dl.iter_batches())
    
        for batch in pbar:
            this_time = time.time() - t0
            # print (this_time, this_time - last_time)
            last_time = this_time

            x, y = batch
            # y = y[:,None]
            y = y.T

            # print ("input", x)
            out = model(x)
            pred = out.features
            # print ("labels", y)
            # print ("pred", pred)
            # print ("out", out)

            # print (n_iter)
            depth_pred = 100*pred[:,0]
            charge_pred = 10000*pred[:,1]

            depth_loss = torch.mean(torch.pow(depth_pred - y[:,0], 2))
            # print ("depth loss", depth_loss)
        
            charge_loss = torch.mean(torch.pow(charge_pred - y[:,1], 2))
            # print ("charge loss", charge_loss)
        
            depth_loss_norm = 1.e-4
            charge_loss_norm = 1.e-10

            # loss = depth_loss*depth_loss_norm + charge_loss*charge_loss_norm
            loss = depth_loss*depth_loss_norm # + charge_loss*charge_loss_norm
            pbarMessage = " ".join(["iter:",
                                    str(n_iter),
                                    "depth loss:",
                                    str(round(depth_loss.item(), 4)),
                                    "charge loss:",
                                    str(round(charge_loss.item(), 4)),
                                    "loss:",
                                    str(round(loss.item(), 4))])
            pbar.set_description(pbarMessage)
            
            # print ("total loss", loss)

            # frac_err = torch.mean(torch.abs((pred - y)/y))

            # loss = torch.mean(torch.pow(pred - y, 2))

            loss_series = torch.cat((loss_series, loss[None].detach()))
            
            # print ("iter", n_iter, "loss", round(loss.item(), 2), round(frac_err.item(), 2))

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            n_iter += 1
            if args.max_iter and n_iter > args.max_iter:
                break
            # if n_iter > iter_per_epoch:
            #     break

        checkpoint_file = os.path.join(args.output_directory,
                                       "checkpoint_"+str(n_epoch)+".ckpt")
        torch.save(dict(model = model.state_dict()), checkpoint_file)
        print (pred, y)
        np.save(loss_log_file, loss_series.detach().cpu().numpy())

    # np.save(args.output_directory, loss_series.detach().cpu().numpy())
        
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

    parser.add_argument('-lr', '--learning_rate', type = float,
                        default = 1.e-4,
                        help = "learning rate for Adam optimizer")
    parser.add_argument('-wd', '--weight_decay', type = float,
                        default = 0,
                        help = "weight decay for Adam optimizer")

    parser.add_argument('-n', '--nEpochs', type = int,
                        default = 5,
                        help = "maximum number of epochs to train")
    parser.add_argument('-m', '--max_iter', type = int,
                        default = None,
                        help = "maximum number of iterations per epoch to train")
    parser.add_argument('-c', '--checkpoint', type = str,
                        default = '.',
                        help = "checkpoint to load")
    parser.add_argument('-f', '--checkpoint_period', type = int,
                        default = 1,
                        help = "save a checkpoint every N epochs")
    parser.add_argument('-o', '--output_directory', type = str,
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
