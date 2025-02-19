from gampixpy.utils import torch_utils
import numpy as np
import os
import h5py
import torch
import MinkowskiEngine as ME

from gampixpy import config, detector, input_parsing

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DataLoader:
    def __init__(self, base_dir, readout_config, batch_size = 4):
        self.base_dir = base_dir
        self.file_list = [os.path.join(base_dir, filename)
                          for filename in os.listdir(base_dir)]

        self.n_files = len(self.file_list)

        self.readout_config = readout_config
        
        self.batch_size = batch_size
        
        self.set_file_load_order()
        
    def set_file_load_order(self):
        self.file_load_order = np.random.choice(self.n_files,
                                                self.n_files,
                                                replace = False)

        return None

    def set_sample_load_order(self):
        n_events = len(np.unique(self.current_file_handle['meta']['event id']))
        
        self.sample_load_order = np.random.choice(n_events,
                                                  n_events,
                                                  replace = False)

        return None
        
    def load_file(self, file_index):
        self.current_file = self.file_list[file_index]
        self.current_file_handle = h5py.File(self.current_file)

        return None

    def load_sample(self, sample_index):
        # tile_st, pixel_st = torch_utils.make_event_sparsetensors(self.current_file_handle,
        #                                                          sample_index,
        #                                                          self.readout_config)
        image = torch_utils.get_event_coo_tensors(self.current_file_handle,
                                                  sample_index,
                                                  self.readout_config)
        label = torch.tensor(torch_utils.get_event_meta(self.current_file_handle,
                                                        sample_index)['primary energy'])

        return image, label

    def iter_batches(self):
        batch_pixel_coords = []
        batch_pixel_feats = []
        # batch_tile_coords = []
        # batch_tile_feats = []

        labels = torch.empty(0,)
        
        for file_index in self.file_load_order:
            self.load_file(file_index)
            self.set_sample_load_order()

            for sample_index in self.sample_load_order:
                sample = self.load_sample(sample_index)
                ((tile_coords, tile_feats), (pixel_coords, pixel_feats)), this_label = sample

                if not torch.any(pixel_coords):
                    continue

                labels = torch.cat((labels, this_label))
                
                batch_pixel_coords.append(pixel_coords)
                batch_pixel_feats.append(pixel_feats.T)
                # batch_tiles_coords.append(tiles_coords)
                # batch_tiles_feats.append(tiles_feats)
    
                if len(batch_pixel_coords) == self.batch_size:
                    pixel_coords_coo, pixel_feats_coo = ME.utils.sparse_collate(batch_pixel_coords,
                                                                                batch_pixel_feats,
                                                                                )
                    pixel_st = ME.SparseTensor(coordinates = pixel_coords_coo.to(device),
                                               features = pixel_feats_coo.to(device))
                    # batch_tiles_st = ME.utils.sparse_collate(batch_tiles_coords,
                    #                                          batch_tiles_feats,
                    #                                          )
                    labels = labels.to(device)

                    yield pixel_st, labels #, batch_tile_st

                    batch_pixel_coords = []
                    batch_pixel_feats = []
                    # batch_tile_coords = []
                    # batch_tile_feats = []

                    labels = torch.empty(0,)

    def __iter__(self):

        for file_index in self.file_load_order:
            self.load_file(file_index)
            self.set_sample_load_order()

            for sample_index in self.sample_load_order:
                yield self.load_sample(sample_index)

class PointSourceLoader:
    def __init__(self,
                 base_dir,
                 readout_config = config.default_readout_params,
                 batch_size = 4):
        self.base_dir = base_dir
        self.file_list = [os.path.join(base_dir, filename)
                          for filename in os.listdir(base_dir)]
        
        self.n_files = len(self.file_list)

        self.readout_config = readout_config

        self.batch_size = batch_size

        self.set_file_load_order()

    def set_file_load_order(self):
        self.file_load_order = np.random.choice(self.n_files,
                                                self.n_files,
                                                replace = False)

        return None

    def set_sample_load_order(self):
        n_events = len(np.unique(self.current_file_handle['meta']['event id']))
        
        self.sample_load_order = np.random.choice(n_events,
                                                  n_events,
                                                  replace = False)

        return None
        
    def load_file(self, file_index):
        self.current_file = self.file_list[file_index]
        self.current_file_handle = h5py.File(self.current_file)

        return None
        
    def load_sample(self, sample_index):

        image = torch_utils.get_event_coo_tensors(self.current_file_handle,
                                                  sample_index,
                                                  self.readout_config,
                                                  # origin = 'coordinate',
                                                  origin = 'edge',
                                                  )
        event_meta = torch_utils.get_event_meta(self.current_file_handle,
                                                sample_index)
        label = torch.tensor(np.array([event_meta['vertex z'],
                                       event_meta['deposited charge'],
                                       ]))

        return image, label

    def iter_batches(self):
        batch_pixel_coords = []
        batch_pixel_feats = []
        # batch_tile_coords = []
        # batch_tile_feats = []

        # labels = torch.empty(2,0)
        labels = None
        
        for file_index in self.file_load_order:
            self.load_file(file_index)
            self.set_sample_load_order()

            for sample_index in self.sample_load_order:
                sample = self.load_sample(sample_index)
                ((tile_coords, tile_feats), (pixel_coords, pixel_feats)), this_label = sample

                if not torch.any(pixel_coords):
                    continue

                # labels = torch.cat((labels, this_label))
                if labels == None:
                    labels = this_label
                else:
                    labels = torch.hstack((labels, this_label))
                
                batch_pixel_coords.append(pixel_coords)
                batch_pixel_feats.append(pixel_feats.T)
                # batch_tiles_coords.append(tiles_coords)
                # batch_tiles_feats.append(tiles_feats)
    
                if len(batch_pixel_coords) == self.batch_size:
                    pixel_coords_coo, pixel_feats_coo = ME.utils.sparse_collate(batch_pixel_coords,
                                                                                batch_pixel_feats,
                                                                                )
                    pixel_st = ME.SparseTensor(coordinates = pixel_coords_coo.to(device),
                                               features = pixel_feats_coo.to(device))
                    # batch_tiles_st = ME.utils.sparse_collate(batch_tiles_coords,
                    #                                          batch_tiles_feats,
                    #                                          )
                    labels = labels.to(device)

                    yield pixel_st, labels #, batch_tile_st

                    batch_pixel_coords = []
                    batch_pixel_feats = []
                    # batch_tile_coords = []
                    # batch_tile_feats = []

                    labels = torch.empty(0,)

    def __iter__(self):

        for file_index in self.file_load_order:
            self.load_file(file_index)
            self.set_sample_load_order()

            for sample_index in self.sample_load_order:
                yield self.load_sample(sample_index)
