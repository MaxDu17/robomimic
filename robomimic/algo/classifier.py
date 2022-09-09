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

@register_algo_factory_func("distance")
def algo_config_to_class(algo_config = None):
    """
    Yields the class for the weighing algorithm. Can be expanded to accomodate more fancier classifiers
    """
    return DistanceWeighter, {}

@register_algo_factory_func("temporal_embedding")
def algo_config_to_class(algo_config = None):
    """
    Yields the class for the weighing algorithm. Can be expanded to accomodate more fancier classifiers
    """
    return TemporalEmbeddingWeighter, {}

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


            losses, accuracy = self._compute_losses(predictions)
            info["accuracy"] = TensorUtils.detach(accuracy)

            info["predictions"] = TensorUtils.detach(predictions["combined"])
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

        # we need to run this through one-shot, or else we risk batchnorm memorizing a pattern
        combined_anchor = {}
        combined_alt = {}
        for key in batch["anchor"].keys():
            # print(key)
            anchor = batch["anchor"][key]
            pos =  batch["positive"][key]
            neg = batch["negative"][key]
            combined_anchor[key] = torch.cat((anchor, anchor), dim=0)
            combined_alt[key] = torch.cat((pos, neg), dim=0)

        self._shuffle(batch)
        for key in batch["anchor"].keys():
            anchor = batch["anchor"][key]
            diff = batch["negative"][key]
            combined_anchor[key] = torch.cat((combined_anchor[key], anchor), dim=0)
            combined_alt[key] = torch.cat((combined_alt[key], diff), dim=0)

        predictions["combined"] = self.nets["policy"](obs_dict_1=combined_anchor, obs_dict_2 = combined_alt)

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
        original_batch_size = predictions["combined"].shape[0] // 3
        assert predictions["combined"].shape[0] % 3 == 0

        pos_target = torch.ones((original_batch_size,1), device = predictions["combined"].device)
        neg_target = torch.zeros((original_batch_size * 2,1), device = predictions["combined"].device)

        concat_target = torch.cat((pos_target, neg_target), dim=0)
        losses["bce_loss"] = self.loss(predictions["combined"], concat_target)

        accuracy = OrderedDict()
        with torch.no_grad():
            hard_labels_pos = (predictions["combined"][:original_batch_size] > 0.9).float()
            accuracy["pos"] = hard_labels_pos.mean()

            hard_labels_neg = (predictions["combined"][original_batch_size:original_batch_size * 2] < 0.1).float()
            accuracy["neg"] = hard_labels_neg.mean()

            hard_labels_diff = (predictions["combined"][2 * original_batch_size:] < 0.1).float()
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
            loss=losses["bce_loss"]
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
        combined_data = {}
        for key in batch["anchor"].keys():
            anchor = batch["anchor"][key]
            pos =  batch["positive"][key]
            neg = batch["negative"][key]
            combined_data[key] = torch.cat((anchor, pos, neg), dim=0)

        embedding = self.nets["policy"](obs=combined_data)["value"]
        return embedding

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
        # same problem

        losses = OrderedDict()
        batch_size = embeddings.shape[0] // 3
        assert embeddings.shape[0] % 3 == 0

        pos_cos_target = torch.ones((batch_size,), device = embeddings.device)
        neg_cos_target = -1 * torch.ones((batch_size,), device = embeddings.device)

        losses["pos"] = self.loss(embeddings[: batch_size], embeddings[batch_size : 2 * batch_size], pos_cos_target)
        losses["neg"] = self.loss(embeddings[: batch_size], embeddings[2 * batch_size : ], neg_cos_target)

        shuffled_negative_indices = torch.randperm(batch_size) + 2 * batch_size

        losses["diff"] = self.loss(embeddings[: batch_size], embeddings[shuffled_negative_indices], neg_cos_target)
        # print(losses)
        accuracy = OrderedDict()
        with torch.no_grad():
            positive_pred = (torch.cosine_similarity(embeddings[: batch_size],
                                                     embeddings[batch_size : 2 * batch_size]) > 0).float()
            accuracy["pos"] = (positive_pred).float().mean()

            negative_pred = (torch.cosine_similarity(embeddings[: batch_size],
                                                     embeddings[2 * batch_size : ]) < 0).float()
            accuracy["neg"] = (negative_pred).float().mean()

            diff_pred = (torch.cosine_similarity(embeddings[: batch_size],
                                                     embeddings[shuffled_negative_indices]) < 0).float()
            accuracy["diff"] = (diff_pred).float().mean()

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
        log["pos_loss"] = info["losses"]["pos"].item()
        log["neg_loss"] = info["losses"]["neg"].item()
        log["diff_loss"] = info["losses"]["diff"].item()
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
        embedding_1 = self.nets["policy"](obs=obs_dict_one["obs"])
        embedding_2 = self.nets["policy"](obs=obs_dict_two["obs"])
        return 0.5 * (torch.cosine_similarity(embedding_1["value"], embedding_2["value"], dim) + 1)

class TemporalEmbeddingWeighter(WeighingAlgo):
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
        observation_with_action = deepcopy(observation_group_shapes)
        observation_with_action["obs"]["actions"] = [7,] #temp hardcoding

        #TODO: verify that it is taking all modalities
        self.nets["embedder"] = MIMO_MLP(
            input_obs_group_shapes=observation_with_action,
            layer_dims=self.algo_config.actor_layer_dims,
            output_shapes=OrderedDict(value=(self.algo_config.embedding_size,)),
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets["future"] = MIMO_MLP(
            input_obs_group_shapes=observation_group_shapes,
            layer_dims=self.algo_config.actor_layer_dims,
            output_shapes=OrderedDict(value=(self.algo_config.embedding_size,)),
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            copy_from = self.nets["embedder"],
            modalities_to_copy = ["agentview_image", "robot0_eye_in_hand_image"] #  WILL NOT WORK WITH LOWDIM!!
        )

        self.nets = self.nets.float().to(self.device)
        self.loss = nn.BCEWithLogitsLoss() #pos_weight = 100 * torch.eye(100, device = self.device) + torch.ones((100,100), device = self.device)) #with logits, or without?

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
        input_batch["future"] = {k: batch["future"][k][:, 0, :] for k in batch["future"]}

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

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

        # from matplotlib import pyplot as plt
        # import numpy as np
        # fig, axs = plt.subplots(ncols=2, nrows = 2)
        # anchor = batch["anchor"]["agentview_image"].cpu().detach().numpy()[14]
        # future = batch["future"]["agentview_image"].cpu().detach().numpy()[14]
        # axs[0, 0].imshow(np.transpose(anchor, (1, 2, 0)))
        # axs[0, 1].imshow(np.transpose(future, (1, 2, 0)))
        # anchor = batch["anchor"]["agentview_image"].cpu().detach().numpy()[12]
        # future = batch["future"]["agentview_image"].cpu().detach().numpy()[12]
        # axs[1, 0].imshow(np.transpose(anchor, (1, 2, 0)))
        # axs[1, 1].imshow(np.transpose(future, (1, 2, 0)))
        # plt.savefig("train.png")
        #
        # import ipdb
        # ipdb.set_trace()

        with TorchUtils.maybe_no_grad(no_grad=validate):
            # this just gets an empty dictionary
            info = super(TemporalEmbeddingWeighter, self).train_on_batch(batch, epoch, validate=validate)

            embeddings = self._forward_training(batch)

            losses, accuracy, sim_matrix = self._compute_losses(embeddings)
            info["accuracy"] = TensorUtils.detach(accuracy)

            # info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)
            info["product_matrix"] = torch.sigmoid(sim_matrix.detach()).cpu().numpy()

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
        combined_data = {}

        current_embedding = self.nets["embedder"](obs=batch["anchor"])["value"]
        future_embedding = self.nets["future"](obs=batch["future"])["value"]
        return current_embedding, future_embedding

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
        # same problem
        current_embedding, future_embedding = embeddings
        batch_size = current_embedding.shape[0]
        # batch x embed dim

        similarity_matrix = current_embedding @ future_embedding.T
        key = torch.eye(batch_size, device = similarity_matrix.device)

        # loss_fn =  nn.BCEWithLogitsLoss(pos_weight = 100 * torch.ones_like(key))
        losses = self.loss(similarity_matrix, key)

        # print(losses)
        accuracy = OrderedDict()
        with torch.no_grad():
            maximums = torch.argmax(similarity_matrix, dim = 1)
            labels = torch.arange(batch_size, device = self.device)
            accuracy["maximums"] = (maximums == labels).float().mean()

            # accuracy["on_diagonal"] = torch.sum(torch.diagonal(similarity_matrix)) / batch_size
            # accuracy["off_diagonal"] = (torch.sum(similarity_matrix) -
            #                             torch.sum(torch.diagonal(similarity_matrix))) / (batch_size**2 - batch_size)

        return losses, accuracy, similarity_matrix

    def _train_step(self, losses):
        """
        Internal helper function for weighting algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        torch.autograd.set_detect_anomaly(True)
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets,
            optim=self.optimizers["embedder"],
            loss=losses,
            # retain_graph = True, #because we need to optimize multiple times
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
        log = super(TemporalEmbeddingWeighter, self).log_info(info)
        log["loss"] = info["losses"].item()
        log["accuracy"] = info["accuracy"]["maximums"].item()
        # log["on_diagonal_average"] = info["accuracy"]["on_diagonal"].item()
        # log["off_diagonal_average"] = info["accuracy"]["off_diagonal"].item()

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
        embedding_1 = self.nets["embedder"](obs=obs_dict_one["obs"])
        embedding_2 = self.nets["embedder"](obs=obs_dict_two["obs"])
        return 0.5 * (torch.cosine_similarity(embedding_1["value"], embedding_2["value"], dim) + 1)

    def compute_embeddings(self, obs_dict):
        assert not self.nets.training
        obs_dict = TensorUtils.to_tensor(obs_dict)
        obs_dict = TensorUtils.to_device(obs_dict, self.device)
        obs_dict = TensorUtils.to_float(obs_dict)
        with torch.no_grad(): #make sure we aren't saving the computation graphs, or we can have a memory leak 
            embed = self.nets["embedder"](obs = obs_dict)["value"].detach().cpu().numpy()
        return embed

class DistanceWeighter(WeighingAlgo):
    """
    A classification model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.MSELoss()

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = WeighterNet.WeighterNet(
            obs_shapes=self.obs_shapes,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            head_activation = "relu",
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
        input_batch["second"] = {k: batch["second"][k][:, 0, :] for k in batch["second"]}
        input_batch["distance"] = batch["distance"]


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
            info = super(DistanceWeighter, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)

            losses, accuracy = self._compute_losses(predictions, batch)
            info["accuracy"] = TensorUtils.detach(accuracy)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)
            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _shuffle(self, batch):
        batch_size = batch["anchor"][list(batch["anchor"].keys())[0]].shape[0]
        permutation = torch.randperm(batch_size)
        batch["negative"] = {key: value[permutation] for key, value in batch["negative"].items()}

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
        predictions = self.nets["policy"](obs_dict_1=batch["anchor"], obs_dict_2=batch["second"])
        # print(predictions[0:10])
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
        labels = torch.unsqueeze(batch["distance"], 1)
        losses["MSE_loss"] = self.loss(predictions, labels)

        with torch.no_grad():
            hard_labels = (torch.abs(predictions - labels) < 3).float()
            accuracy = hard_labels.mean()

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
            loss=losses["MSE_loss"]
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
        log = super(DistanceWeighter, self).log_info(info)
        log["loss"] = info["losses"]["MSE_loss"].item()
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

        distance = self.nets["policy"](obs_dict_one, obs_dict_two)
        raise Exception("can't use distance here")
        return distance