from collections import OrderedDict

import robomimic.models.base_nets as BaseNets
import robomimic.models.weighter_nets as WeighterNet
from robomimic.models.obs_nets import MIMO_MLP, RNN_MIMO_MLP
from robomimic.algo import register_algo_factory_func, WeighingAlgo

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


@register_algo_factory_func("weight")
def algo_config_to_class(algo_config = None):
    """
    Yields the class for the weighing algorithm. Can be expanded to accomodate more fancier classifiers
    """
    return VanillaWeighter, {}

@register_algo_factory_func("weight_contrastive")
def algo_config_to_class(algo_config = None):
    """
    Yields the class for the weighing algorithm. Can be expanded to accomodate more fancier classifiers
    """
    return ContrastiveWeighter, {}

class VanillaWeighter(WeighingAlgo):
    """
    A classification model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.BCELoss()

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        self.nets = nn.ModuleDict()
        self.nets["policy"] = WeighterNet.WeighterNet(
            obs_shapes = self.obs_shapes,
            weight_bounds = None, # self.algo_config.value_bounds,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
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
        input_batch = dict()
        input_batch["obs_1"] = {k: batch["obs_1"][k][:, 0, :] for k in batch["obs_1"]}
        input_batch["obs_2"] = {k: batch["obs_2"][k][:, 0, :] for k in batch["obs_2"]}
        input_batch["label"] = batch["label"]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

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
            info = super(VanillaWeighter, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)

            losses, accuracy = self._compute_losses(predictions, labels)
            info["accuracy"] = TensorUtils.detach(accuracy)

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
        predictions = OrderedDict()

        weights = self.nets["policy"](obs_dict_1=batch["obs_1"], obs_dict_2 = batch["obs_2"])
        predictions["label"] = weights
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
        s_target = torch.unsqueeze(batch["label"], dim = 1) #to match shapes
        similarity = predictions["label"]
        losses["BCE_loss"] = self.loss(similarity, s_target)
        with torch.no_grad():
            hard_labels = (similarity > 0.5).float()
            accuracy = (hard_labels == s_target).float().sum()
        return losses, accuracy


    def _train_step(self, losses):
        """
        Internal helper function for weighting algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["BCE_loss"],
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
        log = super(VanillaWeighter, self).log_info(info)
        log["Loss"] = info["losses"]["BCE_loss"].item()
        if "accuracy" in info:
            log["accuracy"] = info["accuracy"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def similarity_score(self, obs_dict_one, obs_dict_two):
        """
        Args:
            obs_dict_one: first set of observations
            obs_dict_two: second set of observations
        :return:
        """
        assert not self.nets.training
        obs_dict_one = TensorUtils.to_tensor(obs_dict_one)
        obs_dict_one = TensorUtils.to_device(obs_dict_one, self.device)
        obs_dict_one = TensorUtils.to_float(obs_dict_one)

        obs_dict_two = TensorUtils.to_tensor(obs_dict_two)
        obs_dict_two = TensorUtils.to_device(obs_dict_two, self.device)
        obs_dict_two = TensorUtils.to_float(obs_dict_two)
        
        return self.nets["policy"](obs_dict_one, obs_dict_two)

class ContrastiveWeighter(WeighingAlgo):
    """
    A classification model
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        self.nets["policy"] = MIMO_MLP(
            input_obs_group_shapes=observation_group_shapes,
            layer_dims=self.algo_config.actor_layer_dims,
            output_shapes = OrderedDict(value = (self.algo_config.embedding_size, )),
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            pretrained_weights=self.algo_config.pretrained_weights,
            lock=self.algo_config.lock_encoder
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
        input_batch_1 = dict()
        input_batch_2 = dict()
        input_batch_1["obs"] = {k: batch["obs_1"][k][:, 0, :] for k in batch["obs_1"]}
        input_batch_2["obs"] = {k: batch["obs_2"][k][:, 0, :] for k in batch["obs_2"]}
        labels = batch["label"]

        return (TensorUtils.to_device(TensorUtils.to_float(input_batch_1), self.device),
               TensorUtils.to_device(TensorUtils.to_float(input_batch_2), self.device),
               TensorUtils.to_device(TensorUtils.to_float(labels), self.device))


    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): tuple of dictionaries with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch_1, batch_2, labels = batch # different batch strcutrue

        with TorchUtils.maybe_no_grad(no_grad=validate):
            # this just gets an empty dictionary
            info = super(ContrastiveWeighter, self).train_on_batch(batch_1, epoch, validate=validate)
            
            predictions = self._forward_training(batch_1, batch_2)

            losses, true_positive, true_negative = self._compute_losses(predictions, labels)
            info["true_positive"] = TensorUtils.detach(true_positive)
            info["true_negative"] = TensorUtils.detach(true_negative)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch_1, batch_2):
        """
        Internal helper function for weighting algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        embedding_1 = self.nets["policy"](obs=batch_1["obs"])
        embedding_2 = self.nets["policy"](obs=batch_2["obs"])


        predictions["embedding_1"] = embedding_1
        predictions["embedding_2"] = embedding_2
        return predictions

    def _compute_losses(self, predictions, labels):
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

        s_target = labels
        cos_target = deepcopy(s_target)
        cos_target[s_target == 0] = -1 # needed transformation for cosine
        s_target = s_target.type(torch.bool)

        embedding_1, embedding_2 = predictions["embedding_1"]["value"], predictions["embedding_2"]["value"]
        # force -1 and 1 labels
        losses["embedding_cosine_loss"] = nn.CosineEmbeddingLoss(margin = -1)(embedding_1, embedding_2, cos_target)
        with torch.no_grad():
            # how many embeddings are less than orthogonal?
            positive_pred = (torch.cosine_similarity(embedding_1, embedding_2) > 0).float()
            accuracy = (positive_pred == s_target).float().mean()
            true_positive = accuracy
            true_negative = 1 - accuracy
        return losses, true_positive, true_negative


    def _train_step(self, losses):
        """
        Internal helper function for weighting algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["embedding_cosine_loss"],
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
        log = super(ContrastiveWeighter, self).log_info(info)
        log["Loss"] = info["losses"]["embedding_cosine_loss"].item()
        if "true_positive" in info:
            log["true_positive"] = info["true_positive"].item()
        if "true_negative" in info:
            log["true_negative"] = info["true_negative"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def similarity_score(self, obs_dict_one, obs_dict_two):
        """
        Args:
            obs_dict_one: first set of observations
            obs_dict_two: second set of observations
        :return:
        """
        assert not self.nets.training
        embedding_1 = self.nets["policy"](obs=batch_1["obs"])
        embedding_2 = self.nets["policy"](obs=batch_2["obs"])
        return 0.5 * (torch.cosine_similarity(embedding_1["value"], embedding_2["value"], dim) + 1)
