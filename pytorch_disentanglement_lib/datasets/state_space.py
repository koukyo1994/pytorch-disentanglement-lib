# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The original file of this is
#
#     https://github.com/google-research/disentanglement_lib/disentanglement_lib/data/ground_truth/util.py
#
# and has been modified by koukyo1994 to add customizability.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from typing import Sequence


class StateSpace(object):
    """
    Base state space with factors split between latent variable and observation
    """
    def __init__(self, factor_sizes: Sequence[int], latent_factor_indices: Sequence[int]):
        self.factor_sizes = factor_sizes
        self.num_factors = len(factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [
            i for i in range(self.num_factors)
            if i not in self.latent_factor_indices
        ]

    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)

    def sample_latent_factors(self, num: int, random_state: np.random.RandomState):
        """
        Sample a batch of the latent factors
        """
        factors = np.zeros(shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num, random_state)
        return factors

    def sample_all_factors(self, latent_factors: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """
        Sample the remaining remaining factors based on the latent factors
        """
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self.latent_factor_indices] = latent_factors
        # Complete all the other factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples, random_state)
        return all_factors

    def _sample_factor(self, i: int, num: int, random_state: np.random.RandomState):
        return random_state.randint(self.factor_sizes[i], size=num)


class RestrictedStateSpace(StateSpace):
    """
    This state space applies following intervention to the latent factors
    - Set range restrictions to the latent factors
    """
    def __init__(self, factor_sizes: Sequence[int], latent_factor_indices: Sequence[int], factors_range: dict):
        super().__init__(factor_sizes, latent_factor_indices)
        self.factors_range = factors_range

    def sample_latent_factors(self, num: int, random_state: np.random.RandomState):
        """
        Sample a batch of the latent factors
        """
        factors = np.zeros(shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        range_restriction_keys = self.factors_range.keys()
        for pos, i in enumerate(self.latent_factor_indices):
            if i not in range_restriction_keys:
                factors[:, pos] = self._sample_factor(i, num, random_state)
            else:
                if isinstance(self.factors_range[i], tuple):
                    factor_range = self.factors_range[i]
                    range_obj = list(range(factor_range[0], factor_range[1], 1))
                elif isinstance(self.factors_range[i], list):
                    range_obj = self.factors_range[i]
                else:
                    raise NotImplementedError
                factors[:, pos] = self._sample_factor_from_range(num, random_state, range_obj)
        return factors

    def _sample_factor_from_range(self, num: int, random_state: np.random.RandomState, range_obj: list):
        return random_state.choice(range_obj, size=num)
