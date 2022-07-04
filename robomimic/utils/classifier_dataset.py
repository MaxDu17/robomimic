"""
This file contains weighted Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files. It inherits directly form SequenceDataset, but it changes the sampling approach
"""
import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.dataset import SequenceDataset

class ClassifierDataset(SequenceDataset):
    def __init__(self,
                hdf5_path,
                obs_keys,
                dataset_keys,
                radius):
        super(ClassifierDataset, self).__init__(hdf5_path, obs_keys, dataset_keys)
        self.radius = radius
    # just overriding the sampling funcitonality
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        assert self.hdf5_cache_mode == "none", "Caching not yet implemented on the classifier dataset"
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]

        #varying positions
        same = np.random.rand() > 0.5 #the label
        offset = np.random.randint(self.radius)

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        viable_sample_size = self._demo_id_to_demo_length[demo_id]
        demo_end_index = demo_start_index + viable_sample_size
        index_in_demo = index - demo_start_index + demo_index_offset
        # note: this logic does not work if the horizons are different lengths!

        if index + offset > demo_end_index:
            second_index = index_in_demo - offset
        elif index - offset < demo_start_index:
            second_index = index_in_demo + offset
        elif np.random.rand() < 0.5: #coin toss if we aren't at the edge
            second_index = index_in_demo - offset
        else:
            second_index = index_in_demo + offset

        #picking between the demos
        if same:
            demo_id_2 = demo_id #same demo
        else:
            reduced_list = self.demos.copy()
            reduced_list.remove(demo_id) # remaining trajectories
            demo_id_2 = np.random.choice(reduced_list)


        data["label"] = same
        data["obs_1"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )
        data["obs_2"] = self.get_obs_sequence_from_demo(
            demo_id_2,
            index_in_demo=second_index,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )
        return data
