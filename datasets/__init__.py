# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
from .dataset import InfiniteDataReader, EpochDataReader

def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed); random.seed(base_seed); torch.manual_seed(base_seed)


def create_dataloader(batch_size: int, 
                      metas_path: str, 
                      num_actions: int,
                      training: bool,
                      action_mode: str,
                      num_workers: int = 4,
                      include_producer_id: bool = False,
                      data_proportions=None,
                      future_index: int = 0,
                      return_future: bool = False,
                      infinite: bool = True,
                      image_height: int = 224,
                      image_width: int = 224,
                      handler_kwargs=None,
                      ):
    dataset_cls = InfiniteDataReader if infinite else EpochDataReader
    return DataLoader(
        dataset_cls(
            metas_path,
            num_actions=num_actions,
            training=training,
            action_mode=action_mode,
            include_producer_id=include_producer_id,
            data_proportions=data_proportions,
            future_index=future_index,
            return_future=return_future,
            image_height=image_height,
            image_width=image_width,
            handler_kwargs=handler_kwargs,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0
    )
