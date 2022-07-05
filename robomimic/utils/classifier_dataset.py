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
from robomimic.utils.dataset import SequenceDataset

class ClassifierDataset(SequenceDataset):

    def __init__(self,
                 hdf5_path,
                 obs_keys,
                 dataset_keys,
                 frame_stack=1,
                 seq_length=1,
                 pad_frame_stack=True,
                 pad_seq_length=True,
                 get_pad_mask=False,
                 goal_mode=None,
                 hdf5_cache_mode=None,
                 hdf5_use_swmr=True,
                 hdf5_normalize_obs=False,
                 filter_by_attribute=None,
                 load_next_obs=True,
                 priority=False,
                 weighting=False,
                 radius = 15):
        super(ClassifierDataset, self).__init__(
                 hdf5_path,
                 obs_keys,
                 dataset_keys,
                 frame_stack,
                 seq_length,
                 pad_frame_stack,
                 pad_seq_length,
                 get_pad_mask,
                 goal_mode,
                 hdf5_cache_mode,
                 hdf5_use_swmr,
                 hdf5_normalize_obs,
                 filter_by_attribute,
                 load_next_obs,
                 priority,
                 weighting,)
        self.radius = radius
    # just overriding the sampling funcitonality
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        assert self.hdf5_cache_mode == "low_dim", "highdim not yet implemented on the classifier dataset"
        #varying positions
        same = np.random.rand() > 0.5 #the label
        offset = np.random.randint(self.radius)

        demo_id = self._index_to_demo_id[index]

        #picking between the demos
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        viable_sample_size = self._demo_id_to_demo_length[demo_id]
        demo_end_index = demo_start_index + viable_sample_size
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)  # only works for one frame stack
        index_in_demo = index - demo_start_index + demo_index_offset

        if np.random.rand() < 0.5:
            # picking a second index within a radius region, call it "same"
            if index + offset > demo_end_index:
                second_index = index_in_demo - offset
            elif index - offset < demo_start_index:
                second_index = index_in_demo + offset
            elif np.random.rand() < 0.5: #coin toss if we aren't at the edge
                second_index = index_in_demo - offset
            else:
                second_index = index_in_demo + offset
        else:
            # sampling outside of the radius regoin if not same
            perturb = np.random.randint(self.radius, viable_sample_size - self.radius)
            second_index = (index_in_demo + perturb) % viable_sample_size

        keys = list(self.dataset_keys)
        data = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple(keys),
            seq_length=self.seq_length
        )

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
            demo_id,
            index_in_demo=second_index,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=1,
            prefix="obs"
        )
        return data


    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = super().__repr__()
        return "CLASSIFIER DATASET \n" + msg
