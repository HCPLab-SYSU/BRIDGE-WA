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

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class WorldTeacherConfig(PretrainedConfig):
    """
    Configuration for WorldTeacher: Wan backbone (TI2V-5B or VACE-R2V) with multi-task
    future image generation + action diffusion.
    """

    model_type = "world_teacher"

    def __init__(
        self,
        # === Wan backbone ===
        wan_model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        wan_backbone: str = "auto",  # "auto" | "ti2v" | "vace_r2v"
        wan_num_frames: int = 11,
        future_index: int = 10,
        wan_height: int = 256,
        wan_width: int = 256,
        wan_tiled: bool = True,
        wan_tile_size: tuple[int, int] = (34, 34),
        wan_tile_stride: tuple[int, int] = (18, 16),
        wan_torch_dtype: str = "bfloat16",
        freeze_dit: bool = False,
        freeze_text_encoder: bool = True,
        freeze_vae: bool = True,
        freeze_image_encoder: bool = True,
        load_image_encoder: bool = False,
        vace_scale: float = 1.0,

        # === Text/image token dims (Wan defaults) ===
        text_dim: int = 4096,
        image_embed_dim: int = 1280,
        latent_dim: Optional[int] = None,

        # === Future prompt conditioning ===
        use_front_latent_token: bool = True,
        use_wrist_token: bool = True,
        wrist_token_mode: str = "cls",  # "cls" | "mean" | "all"
        use_future_token: bool = True,
        future_prompt_source: str = "pred",  # "pred" | "gt" | "none"
        detach_future_prompt: bool = False,
        joint_single_forward: bool = False,
        sparse_t2v: bool = False,
        delta_t_scale: float = 10.0,

        # === Action diffusion head ===
        action_head_type: str = "dit",  # "dit" | "decoder" | "vision_action"
        vision_action_model_id: str = "./checkpoints/vision_action_pretrain",
        vision_action_language_max_length: int = 50,
        vision_action_use_future_latent: bool = True,
        vision_action_domain_id: int = 0,
        action_hidden_size: int = 1024,
        action_num_layers: int = 6,
        action_num_heads: int = 16,
        action_ffn_dim: int = 4096,
        action_dropout: float = 0.1,
        action_time_dim: int = 128,
        action_latent_hw: int = 2,

        # === Action & proprio ===
        max_action_dim: int = 20,
        real_action_dim: int = 20,
        num_actions: int = 10,
        action_mode: str = "ee6d",
        use_proprio: bool = True,

        # === Loss weights ===
        action_loss_weight: float = 1.0,
        action_supervised_weight: float = 0.0,
        image_loss_weight: float = 1.0,
        recon_loss_weight: float = 0.0,
        image_only_training: bool = False,

        # === Diffusion schedule ===
        image_num_train_timesteps: int = 1000,

        **kwargs,
    ):
        self.wan_model_id = wan_model_id
        self.wan_backbone = wan_backbone
        self.wan_num_frames = wan_num_frames
        self.future_index = future_index
        self.wan_height = wan_height
        self.wan_width = wan_width
        self.wan_tiled = wan_tiled
        self.wan_tile_size = wan_tile_size
        self.wan_tile_stride = wan_tile_stride
        self.wan_torch_dtype = wan_torch_dtype
        self.freeze_dit = freeze_dit
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_vae = freeze_vae
        self.freeze_image_encoder = freeze_image_encoder
        self.load_image_encoder = load_image_encoder
        self.vace_scale = vace_scale

        self.text_dim = text_dim
        self.image_embed_dim = image_embed_dim
        if latent_dim is None:
            backbone = (wan_backbone or "auto").lower()
            if backbone == "auto":
                use_vace = "vace" in (wan_model_id or "").lower()
            else:
                use_vace = backbone in ("vace", "vace_r2v", "r2v", "vace-r2v")
            latent_dim = 16 if use_vace else 48
        self.latent_dim = latent_dim

        self.use_front_latent_token = use_front_latent_token
        self.use_wrist_token = use_wrist_token
        self.wrist_token_mode = wrist_token_mode
        self.use_future_token = use_future_token
        self.future_prompt_source = future_prompt_source
        self.detach_future_prompt = detach_future_prompt
        self.joint_single_forward = joint_single_forward
        self.sparse_t2v = sparse_t2v
        self.delta_t_scale = delta_t_scale

        self.action_hidden_size = action_hidden_size
        self.action_num_layers = action_num_layers
        self.action_num_heads = action_num_heads
        self.action_ffn_dim = action_ffn_dim
        self.action_dropout = action_dropout
        self.action_time_dim = action_time_dim
        self.action_latent_hw = action_latent_hw
        self.action_head_type = action_head_type
        self.vision_action_model_id = vision_action_model_id
        self.vision_action_language_max_length = vision_action_language_max_length
        self.vision_action_use_future_latent = vision_action_use_future_latent
        self.vision_action_domain_id = vision_action_domain_id

        self.max_action_dim = max_action_dim
        self.real_action_dim = real_action_dim
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio

        self.action_loss_weight = action_loss_weight
        self.action_supervised_weight = action_supervised_weight
        self.image_loss_weight = image_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.image_only_training = image_only_training

        self.image_num_train_timesteps = image_num_train_timesteps

        super().__init__(**kwargs)
