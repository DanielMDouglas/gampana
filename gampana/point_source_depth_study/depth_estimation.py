import numpy as np
import matplotlib.pyplot as plt

import h5py
from gampixpy import config

def profile_resolution(estimator, quantity,
                       plot_name = 'resolution.png',
                       outfile_name = 'resolution.dat',
                       binedges_truth = None,
                       xlabel = r'True parameter value',
                       ylabel_up = r'Estimated parameter value $\pm 34\%$',
                       ylabel_lo = r'$\hat{\theta} - \theta$'):
    # print (max(estimator), max(quantity))
    estimator_min = min(estimator)
    # estimator_min = 0
    estimator_max = max(estimator)
    # estimator_max = 3*pitch**2
    # quantity_min = 10
    # quantity_min = min(quantity)
    # # quantity_max = 100
    # quantity_max = max(quantity)

    binedges_estimator = np.linspace(estimator_min, estimator_max, 26)
    bincenters_estimator = 0.5*(binedges_estimator[1:] + binedges_estimator[:-1])
    # binedges_truth = np.linspace(quantity_min, quantity_max, 26)
    bincenters_truth = 0.5*(binedges_truth[1:] + binedges_truth[:-1])
    
    hist, xbins, ybins = np.histogram2d(estimator, quantity,
                                        bins = (binedges_estimator,
                                                binedges_truth),
                                        )

    estimate_distribution = np.matmul(hist.T/np.sum(hist.T, axis = 0), hist/np.sum(hist, axis = 0))
    
    # fig = plt.figure()
    # plt.imshow(estimate_distribution,
    #            origin = 'lower')
    # plt.colorbar()

    cum_distribution = np.cumsum(estimate_distribution, axis = 0)
    quantiles = np.array([np.interp([0.16, 0.5, 0.84], cum_distribution[:,i], bincenters_truth) for i in range(bincenters_truth.shape[0])])

    fig, (ax_up, ax_lo) = plt.subplots(2, 1, sharex=True,
                                       height_ratios = [0.6, 0.4])
    ax_up.errorbar(bincenters_truth,
                   quantiles[:,1],
                   yerr = (quantiles[:,1] - quantiles[:,0],
                           quantiles[:,2] - quantiles[:,1]),
                   fmt = '+',
                   )
    ax_up.plot(np.linspace(binedges_truth[0],
                           binedges_truth[-1],
                           2),
               np.linspace(binedges_truth[0],
                           binedges_truth[-1],
                           2),
               ls = '--',
               color = 'red'),

    ax_up.set_ylabel(ylabel_up)
    
    ax_lo.errorbar(bincenters_truth,
                   quantiles[:,1] - bincenters_truth,
                   yerr = ((quantiles[:,1] - quantiles[:,0]),
                           (quantiles[:,2] - quantiles[:,1])),
                   fmt = '+',
                   )
    ax_lo.plot(np.linspace(binedges_truth[0],
                           binedges_truth[-1],
                           2),
               np.zeros(2),
               ls = '--',
               color = 'red'),

    ax_lo.set_xlabel(xlabel)
    ax_lo.set_ylabel(ylabel_lo)

    plt.savefig(plot_name)

    np.save(outfile_name, np.concatenate((bincenters_truth[:,None], quantiles), axis = 1))

def main(args):
    if args.readout_config == "":
        readout_config = config.default_readout_params
    else:
        readout_config = config.ReadoutConfig(args.readout_config)

    # # input_file = '../../test_inputs/gampixD_point_source.h5'
    # # input_file = '../../test_inputs/superfine_point_source.h5'
    # # input_file = '../../test_inputs/point_source_fine_pitch_1mm.h5'
    # # input_file = '../../test_inputs/point_source_fine_pitch_2mm.h5'
    # # input_file = '../../test_inputs/point_source_fine_pitch_4mm.h5'
    # input_file = '../../test_inputs/point_source_fine_pitch_5mm.h5'

    pitch = readout_config['pixels']['pitch']

    true_x = []
    true_y = []
    true_depth = []
    n_hits = []

    mean_x = []
    mean_y = []
    mean_z = []

    var_x = []
    var_y = []
    var_z = []

    print (args.input)

    for this_input_filename in args.input:
        print ("loading file", this_input_filename)
        f = h5py.File(this_input_filename)
        # print (f.keys())

        for i in np.unique(f['meta']['event id']):
            print ("event", i)
            event_meta_mask = f['meta']['event id'] == i
            event_meta = f['meta'][event_meta_mask]
            
            event_pixel_hits_mask = f['pixel_hits']['event id'] == i
            event_pixel_hits = f['pixel_hits'][event_pixel_hits_mask]
            
            if event_pixel_hits.shape[0] == 0:
                continue

            event_total_charge = np.sum(event_pixel_hits['hit charge'])
            event_mean_x = np.sum(event_pixel_hits['hit charge']*event_pixel_hits['pixel x'])/event_total_charge
            event_mean_y = np.sum(event_pixel_hits['hit charge']*event_pixel_hits['pixel y'])/event_total_charge
            event_mean_z = np.sum(event_pixel_hits['hit charge']*event_pixel_hits['hit z'])/event_total_charge
            
            mean_x.append(event_mean_x)
            mean_y.append(event_mean_y)
            mean_z.append(event_mean_z)
        
            event_var_x = np.sum(event_pixel_hits['hit charge']*event_pixel_hits['pixel x']**2)/event_total_charge - event_mean_x**2
            event_var_y = np.sum(event_pixel_hits['hit charge']*event_pixel_hits['pixel y']**2)/event_total_charge - event_mean_y**2
            event_var_z = np.sum(event_pixel_hits['hit charge']*event_pixel_hits['hit z']**2)/event_total_charge - event_mean_z**2
        
            var_x.append(event_var_x)
            var_y.append(event_var_y)
            var_z.append(event_var_z)
        
            if np.isnan(event_var_x):
                print ("NAN", event_pixel_hits, event_meta)
                
            true_x.append(event_meta['vertex x'][0])
            true_y.append(event_meta['vertex y'][0])
            true_depth.append(event_meta['vertex z'][0])
            n_hits.append(len(event_pixel_hits))

    true_x = np.array(true_x)
    true_y = np.array(true_y)
    true_depth = np.array(true_depth)

    mean_x = np.array(mean_x)
    mean_y = np.array(mean_y)
    mean_z = np.array(mean_z)

    var_x = np.array(var_x)
    var_y = np.array(var_y)
    var_z = np.array(var_z)
    
    # err_x = mean_x - true_x
    # err_y = mean_y - true_y
    # err_z = mean_z - true_depth
    
    # err_mag = np.sqrt(np.pow(err_x, 2) + np.pow(err_y, 2) + np.pow(err_z, 2))

    # plots of the 2D hists of the two proxy variables
    # fig = plt.figure()
    # plt.hist2d(true_depth,
    #            var_x + var_y,
    #            bins = (np.linspace(10, 100, 50),
    #                    np.linspace(0, 3*pitch**2, 50)),
    #                    # np.linspace(0, 0.15, 50)),
    #                    )
    # plt.xlabel(r'True Point depth [cm]')
    # plt.ylabel(r'Estimated depth [cm]')
    
    # fig = plt.figure()
    # plt.hist2d(true_x%pitch,
    #            mean_x%pitch,
    #            bins = (np.linspace(0, pitch, 50),
    #                    np.linspace(0, pitch, 50)),
    #                    )
    # plt.xlabel(r'True Point Source $x$-position [cm]')
    # plt.ylabel(r'Estimated $x$-position [cm]')
    # plt.show()
    
    profile_resolution(var_x + var_y, true_depth,
                       plot_name = args.output + 'depth_resolution.png',
                       outfile_name = args.output + 'depth_resolution.dat',
                       binedges_truth = np.linspace(10, 100, 26),
                       xlabel = r'True Depth [cm]',
                       ylabel_lo = r'$\hat{z} - z$',
                       ylabel_up = r'Estimated Depth [cm]',
                       )
    profile_resolution(mean_x%pitch, true_x%pitch,
                       plot_name = args.output + 'x_resolution.png',
                       outfile_name = args.output + 'x_resolution.dat',
                       binedges_truth = np.linspace(0, pitch, 26),
                       xlabel = r'True $x$-position [cm]',
                       ylabel_lo = r'$\hat{x} - x$',
                       ylabel_up = r'Estimated $x$-position [cm]',
                       )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str,
                        # default = ,
                        # required = True,
                        nargs = '+',
                        help = "input gampix sim output (hdf5)")

    parser.add_argument('-o', '--output', type = str,
                        default = './',
                        help = "output prefix")
    parser.add_argument('-r', '--readout_config',
                        type = str,
                        default = "",
                        help = 'readout configuration yaml')
    
    args = parser.parse_args()

    main(args)
