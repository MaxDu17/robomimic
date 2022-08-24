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
        input_batch["anchor"] = {k: batch["anchor"][k][:, 0, :] for k in batch["anchor"]}
        input_batch["positive"] = {k: batch["positive"][k][:, 0, :] for k in batch["positive"]}
        input_batch["negative"] = {k: batch["negative"][k][:, 0, :] for k in batch["negative"]}

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

            # from matplotlib import pyplot as plt
            # import numpy as np
            # anchor = batch["anchor"]["agentview_image"].cpu().detach().numpy()[12]
            # negative = batch["negative"]["agentview_image"].cpu().detach().numpy()[12]
            # positive = batch["positive"]["agentview_image"].cpu().detach().numpy()[12]
            # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
            # ax1.imshow(np.transpose(anchor, (1, 2, 0)))
            # ax2.imshow(np.transpose(negative, (1, 2, 0)))
            # ax3.imshow(np.transpose(positive, (1, 2, 0)))
            # plt.savefig("train.png")
            # import ipdb
            # ipdb.set_trace()
            # print(batch["anchor"]["agentview_image"].shape)
            # if validate:
            #     from matplotlib import pyplot as plt
            #     import numpy as np
            #     anchor = batch["anchor"]["agentview_image"].cpu().detach().numpy()[12]
            #     negative = batch["negative"]["agentview_image"].cpu().detach().numpy()[12]
            #     positive = batch["positive"]["agentview_image"].cpu().detach().numpy()[12]
            #     fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3)
            #     ax1.imshow(np.transpose(anchor, (1, 2, 0)))
            #     ax2.imshow(np.transpose(negative, (1, 2, 0)))
            #     ax3.imshow(np.transpose(positive, (1, 2, 0)))
            #     plt.savefig("validation.png")
            #     import ipdb
            #     ipdb.set_trace()

            losses, accuracy = self._compute_losses(predictions)
            info["accuracy"] = TensorUtils.detach(accuracy)

            # NOT DONE
            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)
            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _shuffle(self, batch):
        batch_size = batch["anchor"][list(batch["anchor"].keys())[0]].shape[0]
        permutation = torch.randperm(batch_size)
        batch["negative"] = {key : value[permutation] for key, value in batch["negative"].items()}

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

        pos_weights = self.nets["policy"](obs_dict_1=batch["anchor"], obs_dict_2 = batch["positive"])
        neg_weights = self.nets["policy"](obs_dict_1=batch["anchor"], obs_dict_2 = batch["positive"]) #intentional bug
        self._shuffle(batch)
        diff_traj_weights = self.nets["policy"](obs_dict_1=batch["anchor"], obs_dict_2 = batch["negative"])

        predictions["pos"] = pos_weights
        predictions["neg"] = neg_weights
        predictions["diff"] = diff_traj_weights
        return predictions

    def _compute_losses(self, predictions):
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
        pos_target = torch.ones_like(predictions["pos"])
        neg_target = torch.zeros_like(predictions["neg"])
        diff_target = torch.zeros_like(predictions["diff"])

        # print(predictions["pos"][0:3] ,predictions["neg"][0:3], predictions["diff"][0:3])

        concat_pred = torch.cat((predictions["pos"], predictions["neg"], predictions["diff"]), dim = 0)
        concat_target = torch.cat((pos_target, neg_target, diff_target), dim = 0)

        losses["bce_loss"] = self.loss(concat_pred, concat_target)

        accuracy = OrderedDict()
        with torch.no_grad():
            hard_labels_pos = (predictions["pos"] > 0.5).float()
            accuracy["pos"] = hard_labels_pos.mean()

            hard_labels_neg = (predictions["neg"] < 0.5).float()
            accuracy["neg"] = hard_labels_neg.mean()

            hard_labels_diff = (predictions["diff"] < 0.5).float()
            accuracy["diff"] = hard_labels_diff.mean()
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
        # print(losses)
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["bce_loss"], # + losses["diff_loss"],
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
        log["bce_loss"] = info["losses"]["bce_loss"].item()
        # log["pos_loss"] = info["losses"]["pos_loss"].item()
        # log["neg_loss"] = info["losses"]["neg_loss"].item()
        # log["diff_loss"] = info["losses"]["diff_loss"].item()

        if "accuracy" in info:
            log["pos_accuracy"] = info["accuracy"]["pos"].item()
            log["neg_accuracy"] = info["accuracy"]["neg"].item()
            log["diff_accuracy"] = info["accuracy"]["diff"].item()
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
        # import ipdb
        # ipdb.set_trace()
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
            # pretrained_weights=self.algo_config.pretrained_weights,
            # lock=self.algo_config.lock_encoder
        )
        self.nets = self.nets.float().to(self.device)
        self.loss = nn.CosineEmbeddingLoss(margin = -1)

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
        input_batch["anchor"] = {k: batch["anchor"][k][:, 0, :] for k in batch["anchor"]}
        input_batch["positive"] = {k: batch["positive"][k][:, 0, :] for k in batch["positive"]}
        input_batch["negative"] = {k: batch["negative"][k][:, 0, :] for k in batch["negative"]}

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        # input_batch_1 = dict()
        # input_batch_2 = dict()
        # input_batch_1["obs"] = {k: batch["obs_1"][k][:, 0, :] for k in batch["obs_1"]}
        # input_batch_2["obs"] = {k: batch["obs_2"][k][:, 0, :] for k in batch["obs_2"]}
        # labels = batch["label"]
        #
        # return (TensorUtils.to_device(TensorUtils.to_float(input_batch_1), self.device),
        #        TensorUtils.to_device(TensorUtils.to_float(input_batch_2), self.device),
        #        TensorUtils.to_device(TensorUtils.to_float(labels), self.device))


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

        # batch_1, batch_2, labels = batch # different batch strcutrue

        with TorchUtils.maybe_no_grad(no_grad=validate):
            # this just gets an empty dictionary
            info = super(ContrastiveWeighter, self).train_on_batch(batch, epoch, validate=validate)
            
            embeddings = self._forward_training(batch)

            losses, accuracy = self._compute_losses(embeddings)
            info["accuracy"] = TensorUtils.detach(accuracy)

            # info["predictions"] = TensorUtils.detach(predictions)
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
        anchor_embedding = self.nets["policy"](obs=batch["anchor"])
        positive_embedding = self.nets["policy"](obs=batch["positive"])
        negative_embedding = self.nets["policy"](obs=batch["negative"])


        predictions["positive_embedding"] = positive_embedding["value"]
        predictions["negative_embedding"] = negative_embedding["value"]
        predictions["anchor_embedding"] = anchor_embedding["value"]
        return predictions

    def _compute_losses(self, embeddings):
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
        batch_size = predictions["anchor_embedding"].shape[0]

        pos_cos_target = torch.ones_like(embeddings["anchor_embedding"])
        neg_cos_target = -1 * torch.ones_like(embeddings["anchor_embedding"])

        losses["pos"] = self.loss(predictions["anchor_embedding"], predictions["positive_embedding"], pos_cos_target)
        losses["neg"] = self.loss(predictions["anchor_embedding"], predictions["negative_embedding"], neg_cos_target)

        shuffled_negative_embedding = predictions["negative_embedding"][torch.randperm(batch_size)]
        import ipdb
        ipdb.set_trace()
        losses["diff"] = self.loss(predictions["anchor_embedding"], shuffled_negative_embedding, neg_cos_target)

        accuracy = OrderedDict()
        with torch.no_grad():
            positive_pred = (torch.cosine_similarity(predictions["anchor_embedding"],
                                                     predictions["positive_embedding"]) > 0).float()
            accuracy["pos"] = (positive_pred == pos_cos_target).float().mean()

            negative_pred = (torch.cosine_similarity(predictions["anchor_embedding"],
                                                     predictions["negative_embedding"]) < 0).float()
            accuracy["neg"] = (negative_pred == neg_cos_target).float().mean()

            diff_pred = (torch.cosine_similarity(predictions["anchor_embedding"],
                                                     shuffledD_negative_embedding) < 0).float()
            accuracy["diff"] = (diff_pred == neg_cos_target).float().mean()

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
            loss=losses["neg"] + losses["pos"] + losses["diff"],
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
        log["pos_loss"] = not info["losses"]["pos"].item()
        log["neg_loss"] = not info["losses"]["neg"].item()
        log["diff_loss"] = not info["losses"]["diff"].item()
        log["pos_accuracy"] = not info["accuracy"]["pos"].item()
        log["neg_accuracy"] = not info["accuracy"]["neg"].item()
        log["diff_accuracy"] = not info["accuracy"]["diff"].item()

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
        embedding_1 = self.nets["policy"](obs=obs_dict_one["obs"])
        embedding_2 = self.nets["policy"](obs=obs_dict_two["obs"])
        return 0.5 * (torch.cosine_similarity(embedding_1["value"], embedding_2["value"], dim) + 1)
