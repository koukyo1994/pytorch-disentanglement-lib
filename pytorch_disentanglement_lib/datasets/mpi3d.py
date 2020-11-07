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
#     https://github.com/google-research/disentanglement_lib/disentanglement_lib/data/ground_truth/mpi3d.py
#
# and has been modified by koukyo1994 to add customizability.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pathlib import Path
from typing import Union

from pytorch_disentanglement_lib.datasets.base import DatasetBase


class MPI3D(DatasetBase):
    """
    MPI3D datasets have been introduced as a part of NeurIPS 2019 Disentanglement Competition.
    (http://www.disentanglement-challenge.com)

    There are three different datasets:
    1. Simplistic rendered images (mpi3d_toy)
    2. Realistic rendered images (mpi3d_realistic)
    3. Real world images (mpi3d_real).

    Currently only mpi3d_toy is publicly available. More details about this dataset can be found in
    'On the Transfer of Inductive Bias from Simulation to the Real World: a New Disentanglement Dataset'
    (https://arxiv.org/abs/1906.03292).

    The ground truth factors of variation in this dataset are:
    0 - Object color (4 different values for the simulated datasets and 6 for the real one)
    1 - Object shape (4 different values for the simulated datasets and 6 for the real one)
    2 - Object size (2 diffrent values)
    3 - Camera height (3 different values)
    4 - Backgound colors (3 different values)
    5 - Firs DOF (40 different values)
    6 - Second DOF (40 different values)
    """

    def __init__(self, state_space, dataset_path: Union[str, Path], mode="mpi3d_toy"):
        data = np.load(dataset_path)
        if mode in ["mpi3d_toy", "mpi3d_realistic"]:
            self.factor_sizes = [4, 4, 2, 3, 3, 40, 40]
        elif mode == "mpi3d_real":
            self.factor_sizes = [6, 6, 2, 3, 3, 40, 40]
        else:
            raise ValueError("Unknown mode provided")

        self.images = data["images"]
        self.latent_factor_indices = [0, 1, 2, 3, 4, 5, 6]
        self.num_total_factors = 7
        self.state_space = state_space(factor_sizes=self.factor_sizes, latent_factor_indices=self.latent_factor_indices)
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [64, 64, 3]

    def sample_factors(self, num: int, random_state: np.random.RandomState):
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors: np.ndarray, random_state: np.random.RandomState):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.images[indices] / 255.
