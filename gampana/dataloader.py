from gampixpy.utils import torch_utils
import numpy as np
import os
import h5py
import torch
import MinkowskiEngine as ME

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
        # (tile_coords, tile_feats), (pixel_coords, pixel_feats) = image

        print ("image tensor", image)
        print ("label tensor", label)
        return image, label

    def iter_batches(self):
        batch_pixel_coords = []
        batch_pixel_feats = []
        # batch_tile_coords = []
        # batch_tile_feats = []
        
        for file_index in self.file_load_order:
            self.load_file(file_index)
            self.set_sample_load_order()

            for sample_index in self.sample_load_order:
                sample = self.load_sample(sample_index)
                ((tile_coords, tile_feats), (pixel_coords, pixel_feats)), label = sample

                print (label)
                
                batch_pixel_coords.append(pixel_coords)
                batch_pixel_feats.append(pixel_feats)
                # batch_tiles_coords.append(tiles_coords)
                # batch_tiles_feats.append(tiles_feats)

                print ("imag pixel feats", len(batch_pixel_coords), pixel_feats)
                    
                if len(batch_pixel_coords) == self.batch_size:
                    batch_pixel_st = ME.utils.sparse_collate(batch_pixel_coords,
                                                             batch_pixel_feats,
                                                             )
                    batch_tiles_st = ME.utils.sparse_collate(batch_tiles_coords,
                                                             batch_tiles_feats,
                                                             )

                    print ("batck pixel st", batch_pixel_st)
                    yield batch_pixel_st #, batch_tile_st

                    batch_pixel_coords = []
                    batch_pixel_feats = []
                    # batch_tile_coords = []
                    # batch_tile_feats = []

    def __iter__(self):

        for file_index in self.file_load_order:
            self.load_file(file_index)
            self.set_sample_load_order()

            for sample_index in self.sample_load_order:
                yield self.load_sample(sample_index)
