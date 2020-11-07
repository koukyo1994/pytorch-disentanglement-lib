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
#     https://github.com/google-research/disentanglement_lib/disentanglement_lib/data/ground_truth/shapes3d.py
#
# and has been modified by koukyo1994 to add customizability.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np

from pathlib import Path
from typing import Union

from pytorch_disentanglement_lib.datasets.base import DatasetBase
from pytorch_disentanglement_lib.datasets.state_space import StateSpace


class Shapes3D(DatasetBase):
    """
    Shapes3D dataset.

    The data set was originally introduced in "Disentangling by Factorising".
    (http://proceedings.mlr.press/v80/kim18b.html)

    Original data source can be found at https://github.com/deepmind/3d-shapes

    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)
    """
    def __init__(self, state_space: StateSpace, dataset_path: Union[str, Path], data_format="h5"):
        assert data_format in {"h5", "npy", "npz"}, \
            "data_format must be one of 'h5', 'npy' and 'npz"

        # In deepmind's repository, the data is distributed in h5 format.
        if data_format == "h5":
            data = h5py.File(dataset_path, "r")
            images = np.array(data["images"])
            labels = data["labels"]
            self.images = images / 255.0
            n_samples = images.shape[0]
        # In disentanglement_lib repository, the dataset is implemented for npz format.
        elif data_format in {"npy", "npz"}:
            data = np.load(dataset_path, encoding="latin1")
            images = data["images"]
            labels = data["labels"]
            n_samples = np.prod(images.shape[0:6])
            self.images = (
                images.reshape([n_samples, 64, 64, 3]).astype(np.float32) / 255.0)
            labels = labels.reshape([n_samples, 6])
        else:
            raise NotImplementedError
        self.factor_sizes = [10, 10, 10, 8, 4, 15]
        self.latent_factor_indices = list(range(6))
        self.num_total_factors = labels.shape[1]
        self.state_space = state_space
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(slef):
        return [64, 64, 3]

    def sample_factors(self, num: int, random_state: np.random.RandomState):
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors: np.ndarray, random_state: np.random.RandomState):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.images[indices]
