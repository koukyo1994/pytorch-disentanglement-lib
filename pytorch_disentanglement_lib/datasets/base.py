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
#     https://github.com/google-research/disentanglement_lib/disentanglement_lib/data/ground_truth/ground_truth_data.py
#
# and has been modified by koukyo1994 to add customizability.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class DatasetBase(object):
    """
    Abstract class for datasets that are two-step generative models
    """
    @property
    def num_factors(self):
        raise NotImplementedError

    @property
    def factors_num_values(self):
        raise NotImplementedError

    @property
    def observation_shape(self):
        raise NotImplementedError

    def sample_factors(self, num: int, random_state: np.random.RandomState):
        """
        Sample a batch of factors Y
        """
        raise NotImplementedError

    def sample_observations_from_factors(self, factors: np.ndarray, random_state: np.random.RandomState):
        """
        Sample a batch of observations X given a batch of factors Y
        """
        raise NotImplementedError

    def sample(self, num: int, random_state: np.random.RandomState):
        """
        Sample a batch of observations X and factors Y
        """
        factors = self.sample_factors(num, random_state)
        observations = self.sample_observations_from_factors(factors, random_state)
        return factors, observations

    def sample_observations(self, num: int, random_state: np.random.RandomState):
        """
        Sample a batch of observations X
        """
        return self.sample(num, random_state)[1]
