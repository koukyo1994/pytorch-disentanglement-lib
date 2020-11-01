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
#     https://github.com/google-research/disentanglement_lib/disentanglement_lib/data/ground_truth/dsprites.py
#
# and has been modified by koukyo1994 to add customizability.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pathlib import Path
from PIL import Image
from typing import Union

from pytorch_disentanglement_lib.datasets.base import DatasetBase
from pytorch_disentanglement_lib.datasets.state_space import StateSpace


class DSprites(DatasetBase):
    """
    DSprites dataset.

    The data set was originally introduced in "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework" and can be downloaded from
    https://github.com/deepmind/dsprites-dataset.
    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """

    def __init__(self, state_space: StateSpace, dataset_path: Union[str, Path], latent_factor_indices=None):
        # By default, all factors (including shape) are considered  as ground truth factors
        if latent_factor_indices is None:
            latent_factor_indices = list(range(6))

        self.latent_factor_indices = latent_factor_indices
        self.data_shape = [64, 64, 1]

        data = np.load(dataset_path, encoding="latin1", allow_pickle=True)
        self.images = np.array(data["imgs"])
        self.factor_sizes = data["metadata"][()]["latents_sizes"][1:]
        self.full_factor_sizes = [3, 6, 40, 32, 32]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.state_space = state_space

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num: int, random_state: np.random.RandomState):
        """
        Sample a batch of factors Y.
        """
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors: np.ndarray, random_state: np.random.RandomState):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)


class ColorDSprites(DSprites):
    """
    Color DSprites.
    This data set is the same as the original DSprites dataset except that when
    sampling the observation X, the sprites is colored in a randomly sampled color.
    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 differnt values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """

    def __init__(self, state_space: StateSpace, dataset_path: Union[str, Path], latent_factor_indices=None):
        super().__init__(state_space, dataset_path, latent_factor_indices)
        self.data_shape = [64, 64, 3]

    def sample_obbservations_from_factors(self, factors: np.ndarray, random_state: np.random.RandomState):
        no_color_observations = super().sample_observations_from_factors(factors, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)
        color = np.repeat(
            np.repeat(
                random_state.uniform(0.5, 1, [observations.shape[0], 1, 1, 3]),
                observations.shape[1],
                axis=1
            ),
            observations.shape[2],
            axis=2
        )
        return observations * color


class NoisyDSprites(DSprites):
    """
    Noisy DSprites.
    This data set is the same as the original DSprites data set except that when
    sampling the observation X, the background pixels are replaced with random noise.
    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 differnt values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """
    def __init__(self, state_space: StateSpace, dataset_path: Union[str, Path], latent_factor_indices=None):
        super().__init__(state_space, dataset_path, latent_factor_indices)
        self.data_shape = [64, 64, 3]

    def sample_observations_from_factors(self, factors: np.ndarray, random_state: np.random.RandomState):
        no_color_observations = super().sample_observations_from_factors(factors, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)
        color = random_state.uniform(0, 1, [observations.shape[0], 64, 64, 3])
        return np.minimum(observations + color, 1.0)


class ScreamDSprites(DSprites):
    """
    Scream DSprites.
    This data set is the same as the original DSprites data set except that when
    sampling the observations X, a random patch of the Scream image is sampled as
    the background and the sprite is embedded into the image by inverting the
    color of the sampled patch at the pixels of the sprite.
    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 differnt values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """

    def __init__(self, state_space: StateSpace, dataset_path: Union[str, Path], scream_path: Union[str, Path], latent_factor_indices=None):
        super().__init__(state_space, dataset_path, latent_factor_indices)
        self.data_shape = [64, 64, 3]

        scream = Image.open(scream_path)
        scream.thumbnail((350, 274))
        self.scream = np.array(scream) * 1. / 255

    def sample_observations_from_factors(self, factors: np.ndarray, random_state: np.random.RandomState):
        no_color_observations = super().sample_observations_from_factors(factors, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)

        for i in range(observations.shape[0]):
            x_crop = random_state.randint(0, self.scream.shape[0] - 64)
            y_crop = random_state.randint(0, self.scream.shape[1] - 64)
            background = (self.scream[x_crop: x_crop + 64, y_crop: y_crop + 64] +
                          random_state.unifor(0, 1, size=3)) / 2.0
            mask = (observations[i] == 1)
            background[mask] = 1 - background[mask]
            observations[i] = background
        return observations
