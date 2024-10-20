import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from rlkit.torch.networks import Mlp, CNN
from rlkit.torch.networks.basic import MultiInputSequential
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)


class TorchStochasticPolicy(
    DistributionGenerator,
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, obs_np, ):
        actions, aux_output = self.get_actions(obs_np[None])
        return actions[0, :], {}, aux_output

    def get_actions(self, obs_np, ):
        dist, aux_output = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        # print("actions in get_actions function: ", actions)
        return elem_or_tuple_to_numpy(actions), elem_or_tuple_to_numpy(aux_output)

    def _get_dist_from_np(self, *args, **kwargs):
        # print("args in obs_np: ", args)
        # print("args len: ", len(args))
        # print("args[0] in obs_np: ", args[0])
        # print("args[0] len: ", len(args[0]))
        # print("args[0][0] in obs_np: ", args[0][0])
        # print("args[0][0] len: ", len(args[0][0]))
        # print("args[0][0][0] in obs_np: ", args[0][0][0])
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        # print("torch_args in obs_np: ", torch_args)
        # print("torch_kwargs in obs_np: ", torch_kwargs)
        dist = self(*torch_args, **torch_kwargs)
        # print("dist: ", dist)
        return dist


class PolicyFromDistributionGenerator(
    MultiInputSequential,
    TorchStochasticPolicy,
):
    """
    Usage:
    ```
    distribution_generator = FancyGenerativeModel()
    policy = PolicyFromBatchDistributionModule(distribution_generator)
    ```
    """
    pass


class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
            self,
            action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator

    def forward(self, *args, **kwargs):
        dist, aux_output = self._action_distribution_generator.forward(
            *args, **kwargs)
        return Delta(dist.mle_estimate()), aux_output
