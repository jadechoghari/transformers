# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert DINOv2Exp checkpoints from the original repository:
URL: https://github.com/facebookresearch/dinov2

"""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision import transforms

from transformers import DINOv2ExpConfig, DINOv2ExpModel, BitImageProcessor
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dinov2_config(model_name):
    config = DINOv2ExpConfig(image_size=518, patch_size=14)

    # size of the architecture
    # change couple of model sizes model comes in
    if "vits" in model_name:
        config.hidden_size = 384
        config.num_attention_heads = 6
    elif "vitb" in model_name:
        pass
    elif "vitl" in model_name:
        config.hidden_size = 1024
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif "vitg" in model_name:
        config.use_swiglu_ffn = True
        config.hidden_size = 1536
        config.num_hidden_layers = 40

    else:
        raise ValueError("Model not supported")

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))


    for i in range(config.num_hidden_layers):

        #layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias"))

        # MLP
        if config.use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"encoder.layer.{i}.mlp.w3.bias"))
        else: 
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias"))

        # layerscale
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1"))

        # attention projection layer
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"encoder.layer.{i}.attention.output.dense.bias"))


        # final layer norm
        rename_keys.append(("norm.weight", "layernorm.weight"))
        rename_keys.append(("norm.bias", "layernorm.bias"))


        # TODO attention



    # fmt: on
    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# def rename_key(dct, old, new):
#     val = dct.pop(old)
#     dct[new] = val
def rename_key(dct, old, new):
    if new not in dct:
        val = dct.pop(old)
        dct[new] = val



# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@torch.no_grad()
def convert_dinov2exp_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DINOv2Exp structure.
    """

    # define default DINOv2Exp configuration
    config = get_dinov2_config(model_name)

    # load original model from torch hub very handy
    original_model = torch.hub.load("facebookresearch/dinov2", model_name)
    original_model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = original_model.state_dict()

    # rename keys to match hf implementation
    rename_keys = create_rename_keys(config)

    print('rename_keys', rename_keys)
    for src, dest in rename_keys:
        # bug to fix
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = DINOv2ExpModel(config, add_pooling_layer=False).eval()
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    assert len(missing_keys) == 0
    assert unexpected_keys == ["mask_token"]
    # print("Missing keys: ", missing_keys)
    # print("Unexpected keys: ", unexpected_keys)

    # Check outputs on an image, prepared by ViTImageProcessor,
    # image_processor = ViTImageProcessor()
    # encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # pixel_values = encoding["pixel_values"]


    # load image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # preprocess image
    transformations = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD ),
        ]
    )

    original_pixel_values = transformations(image).unsqueeze(0)

    #always check the size (can be different)
    processor = BitImageProcessor(do_resize=True, 
                                  resample=PILImageResampling.BICUBIC,
                                  size={"shortest_edge": 256},
                                do_center_crop=True, crop_size={"height": 224, "width": 224},
                                do_normalize=True,
                                image_mean=IMAGENET_DEFAULT_MEAN, 
                                image_std=IMAGENET_DEFAULT_STD)
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # TODO verify pixel values
    # assert torch.allclose(pixel_values, original_pixel_values, atol=1e-4)

    with torch.no_grad():
        outputs = model(original_pixel_values)
        original_outputs = original_model(pixel_values)

    # last_hidden_state = outputs.last_hidden_state
    # print("Outputs:", last_hidden_state.shape)
    # print("First values of final hidden states:", last_hidden_state[0, :3, :3])
    # TODO assert values

    # add assert for a value for some output to compare
    # use torch.allclose

    # assert torch.allclose(final_hidden_state_cls_token, outputs.last_hidden_state[:, 0, :], atol=1e-1)
    # for default model uncomment this
    # expected_slice = torch.tensor([[-2.1849, -0.3433,  1.0913],
    #     [-3.2696, -0.7386, -0.8044],
    #     [-3.0603,  1.2498, -0.7685]])
    # assert torch.allclose(last_hidden_state[0, :3, :3], expected_slice, atol=1e-4)


    # for vits model here we go:
    assert outputs.last_hidden_state[:, 0].shape == original_outputs.shape 
    assert torch.allclose(outputs.last_hidden_state[:, 0], original_outputs, atol=1e-5) #was 1e-5
    print("Looks ok!")


    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # print(f"Saving image processor to {pytorch_dump_folder_path}")
        # image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_name_to_hf_name = {
            "dinov2_vits14": "dinov2-small",
            "dinov2_vitb14": "dinov2-base",
            "dinov2_vitl14": "dinov2-large",
            "dinov2_vitg14": "dinov2-giant",
            "dinov2_vits14_1layer": "dinov2-small-imagenet1k-1-layer",
            "dinov2_vitb14_1layer": "dinov2-base-imagenet1k-1-layer",
            "dinov2_vitl14_1layer": "dinov2-large-imagenet1k-1-layer",
            "dinov2_vitg14_1layer": "dinov2-giant-imagenet1k-1-layer",
        }
        name = model_name_to_hf_name[model_name]
        model.push_to_hub(f"jadechoghari/{name}")
        # image_processor.push_to_hub(f"jadechoghari/{model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dinov2_vitb14",  # mind need to change name to exp!
        type=str,
        choices=["dinov2_vitb14", "dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_dinov2exp_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)

