from collections import OrderedDict

import robomimic.models.base_nets as BaseNets
import robomimic.models.classifier_net as ClassifierNet
import robomimic.models.vae_nets as VAENet
from robomimic.models.obs_nets import MIMO_MLP, RNN_MIMO_MLP
from robomimic.algo import register_algo_factory_func, TimeToSuccessAlgo

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

@register_algo_factory_func("timesuccess")
def algo_config_to_class(algo_config = None):
    """
    Yields the class for the weighing algorithm. Can be expanded to accomodate more fancier classifiers
    """
    return TimeToSuccess, {}

class TimeToSuccess(TimeToSuccessAlgo):
    """
    A classification model for time to success
    """

    def __init__(self, bin_segmentation = [0, 5, 10, 50, 100, 401], **kwargs):
        self.loss = nn.CrossEntropyLoss()
        self.bin_segmentation = bin_segmentation
        super().__init__(**kwargs)

    def _bin_to_time(self, bin):
        # THIS FUNCTION CAN CHANGE
        return self.bin_segmentation[bin + 1] #simplest algorithm: find upper bound

    def _index_to_bin(self, indices):
        # compute [batch] regressions to bin for one-hot keys

        for i in range(len(self.bin_segmentation) - 1):
            range_low = self.bin_segmentation[i]
            range_high = self.bin_segmentation[i + 1]
            indices[torch.logical_and(indices > range_low, indices <= range_high)] = i #stick with the lower half
        return indices

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = ClassifierNet.ClassifierNet(
            obs_shapes=self.obs_shapes,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            n_bins = len(self.bin_segmentation) - 1,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(TimeToSuccess, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)

            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)
            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info


    def _forward_training(self, batch):
        """
        Internal helper function for weighting algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = self.nets["policy"](obs_dict=batch["obs"])
        return predictions


    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for weighting algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        labels = self._index_to_bin(batch["time_to_success"]).long()
        losses["CrossEntropy"] = self.loss(predictions, labels)

        return losses

    def _train_step(self, losses):
        """
        Internal helper function for weighting algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        # print(losses)
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["CrossEntropy"]
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(TimeToSuccess, self).log_info(info)
        log["loss"] = info["losses"]["CrossEntropy"].item()
        if "accuracy" in info:
            log["accuracy"] = info["accuracy"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def time_to_success(self, obs):
        assert not self.nets.training
        obs_dict = TensorUtils.to_tensor(obs_dict)
        obs_dict = TensorUtils.to_device(obs_dict, self.device)
        obs_dict = TensorUtils.to_float(obs_dict)
        distr = self.nets["policy"](obs_dict)
        import ipdb
        ipdb.set_trace()
        bin = torch.argmax(distr).item()
        return self._bin_to_time(bin)
