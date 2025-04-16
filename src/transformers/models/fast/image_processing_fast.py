#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/fast/modular_fast.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_fast.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2025 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging
from ...utils.import_utils import is_cv2_available, is_scipy_available, is_torch_available, requires_backends


if is_cv2_available():
    import cv2

if is_scipy_available():
    import scipy.ndimage as ndi
    from scipy.spatial import ConvexHull

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


def connected_components(image, connectivity=8):
    """
    Computes connected components of a binary image using SciPy.

    Parameters:
        image (np.ndarray): Binary input image (0s and 1s)
        connectivity (int): Connectivity, 4 or 8 (default is 8)

    Returns:
        labels (np.ndarray): Labeled output image
        num_labels (int): Number of labels found
    """
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)  # 4-connectivity

    labels, num_labels = ndi.label(image, structure=structure)
    return num_labels, labels


def compute_min_area_rect(points):
    """
    Compute the minimum area rotated bounding rectangle around a set of 2D points.

    Args:
        points (np.ndarray): Nx2 array of (x, y) coordinates.

    Returns:
        tuple: ((cx, cy), (w, h), angle) where
            - (cx, cy) is the center of the rectangle,
            - (w, h) are the width and height of the rectangle,
            - angle is the rotation angle in degrees.
    """
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    edges = np.diff(hull_points, axis=0, append=hull_points[:1])
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
    edge_angles = np.unique(edge_angles)

    min_area = float("inf")
    best_box = None

    for angle in edge_angles:
        # Rotate points by -angle (clockwise)
        R = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        rotated = points @ R.T

        # Bounding box in rotated space
        min_x, min_y = rotated.min(axis=0)
        max_x, max_y = rotated.max(axis=0)
        w = max_x - min_x
        h = max_y - min_y
        area = w * h

        if area < min_area:
            min_area = area
            best_box = (min_x, min_y, max_x, max_y, angle, w, h)

    min_x, min_y, max_x, max_y, angle, w, h = best_box
    center_rotated = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    R_inv = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = center_rotated @ R_inv.T

    angle_deg = np.degrees(angle)

    # we ensure angle is in the range [-90, 0)
    while angle_deg >= 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180

    return (tuple(center), (w, h), angle_deg)


def get_box_points(rect):
    """
    Computes the four corner points of a rotated rectangle in OpenCV's order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]

    Args:
        rect (tuple): ((cx, cy), (w, h), angle)
                      - Center coordinates (cx, cy)
                      - Width and height (w, h)
                      - Rotation angle in degrees

    Returns:
        np.ndarray: (4, 2) array of corner points in OpenCV order.
    """
    (center_x, center_y), (width, height), angle_degrees = rect
    angle_radians = np.radians(angle_degrees)

    cos_angle = np.cos(angle_radians) * 0.5
    sin_angle = np.sin(angle_radians) * 0.5

    # compute top-left and top-right corners
    top_left_x = center_x - sin_angle * height - cos_angle * width
    top_left_y = center_y + cos_angle * height - sin_angle * width
    top_left = [top_left_x, top_left_y]

    top_right_x = center_x + sin_angle * height - cos_angle * width
    top_right_y = center_y - cos_angle * height - sin_angle * width
    top_right = [top_right_x, top_right_y]

    # mirror across the center to get the other two corners
    bottom_right = [2 * center_x - top_left_x, 2 * center_y - top_left_y]
    bottom_left = [2 * center_x - top_right_x, 2 * center_y - top_right_y]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


class FastImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Fast image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 640}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        size_divisor (`int`, *optional*, defaults to 32):
            Ensures height and width are rounded to a multiple of this value after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.

        min_area (`int`, *optional*, defaults to 250):
            Minimum area (in pixels) for a region to be considered a valid detection.
            Regions smaller than this threshold will be ignored during post-processing.
        pooling_size (`int`, *optional*, defaults to 9):
            Size of the pooling window used during region proposal aggregation or feature map downsampling.
            This controls the granularity of spatial features extracted from the image.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        size_divisor: int = 32,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_center_crop: bool = False,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = IMAGENET_DEFAULT_MEAN,
        image_std: Optional[Union[float, List[float]]] = IMAGENET_DEFAULT_STD,
        do_convert_rgb: bool = True,
        min_area: int = 250,
        pooling_size: int = 9,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 640}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_convert_rgb = do_convert_rgb

        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "size_divisor",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
        self.min_area = min_area
        self.pooling_size = pooling_size

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"] , with the longest edge
        resized to keep the input aspect ratio. Both the height and width are resized to be divisible by 32.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            size_divisor (`int`, *optional*, defaults to `32`):
                Ensures height and width are rounded to a multiple of this value after resizing.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            default_to_square (`bool`, *optional*, defaults to `False`):
                The value to be passed to `get_size_dict` as `default_to_square` when computing the image size. If the
                `size` argument in `get_size_dict` is an `int`, it determines whether to default to a square image or
                not.Note that this attribute is not used in computing `crop_size` via calling `get_size_dict`.
        """
        if "shortest_edge" in size:
            size = size["shortest_edge"]
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        height, width = get_resize_output_image_size(
            image, size=size, input_data_format=input_data_format, default_to_square=False
        )
        if height % self.size_divisor != 0:
            height += self.size_divisor - (height % self.size_divisor)
        if width % self.size_divisor != 0:
            width += self.size_divisor - (width % self.size_divisor)

        return resize(
            image,
            size=(height, width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        size_divisor: Optional[int] = None,
        resample: PILImageResampling = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[int] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            size_divisor (`int`, *optional*, defaults to `32`):
                Ensures height and width are rounded to a multiple of this value after resizing.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        all_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

            if do_center_crop:
                image = self.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            all_images.append(image)
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            for image in all_images
        ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def _max_pooling(self, input_tensor, scale=1):
        kernel_size = self.pooling_size // 2 + 1 if scale == 2 else self.pooling_size
        padding = (self.pooling_size // 2) // 2 if scale == 2 else (self.pooling_size - 1) // 2

        pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)

        pooled_output = pooling(input_tensor)
        return pooled_output

    def post_process_text_detection(self, output, target_sizes=None, threshold=0.5, output_type="boxes"):
        """
        Post-processes the raw model output to generate bounding boxes and scores for text detection.

        Args:
            output (dict): Dictionary containing model outputs. Must include key `"logits"` (Tensor of shape [B, C, H, W]).
            target_sizes (List[Tuple[int, int]], optional): Original image sizes (height, width) for each item in the batch.
                                                            Used to scale detection results back to original image dimensions.
            threshold (float): Confidence threshold for filtering low-score text regions.
            output_type (str): "boxes" (rotated rectangles) or "polygons" (polygon).

        Returns:
            List[Dict]: Each dict contains:
                - "boxes": np.ndarray of shape (N, 5) if output_type="boxes", or (N, 8) if output_type="polygons"
                - "scores": np.ndarray of shape (N,)
        """
        if output_type not in ["boxes", "polygons"]:
            raise ValueError(f"Invalid output_type: {output_type}. Must be 'boxes' or 'polygons'.")
        scale = 2
        out = output["logits"]
        batch_size, _, H, W = out.shape

        # generate score maps
        texts = F.interpolate(out[:, 0:1, :, :], size=(H, W), mode="nearest")
        texts = self._max_pooling(texts, scale=scale)
        score_maps = torch.sigmoid(texts)
        score_maps = score_maps.squeeze(1)

        # generate label maps
        kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B x H x W
        labels_ = []
        for kernel in kernels.cpu().numpy():
            _, label_ = connected_components(kernel)
            labels_.append(label_)
        labels_ = torch.from_numpy(np.array(labels_)).unsqueeze(1).float()
        labels = self._max_pooling(labels_, scale=scale).squeeze(1).to(torch.int32)

        results = []
        for i in range(batch_size):
            if target_sizes is not None:
                orig_h, orig_w = target_sizes[i]
                scale_x = orig_w / W
                scale_y = orig_h / H
            else:
                scale_x = scale_y = 1.0

            keys = torch.unique(labels_[i], sorted=True)
            if output_type == "boxes":
                bboxes, scores = self._get_rotated_boxes(keys, labels[i], score_maps[i], (scale_x, scale_y), threshold)
            elif output_type == "polygons":
                bboxes, scores = self._get_polygons(keys, labels[i], score_maps[i], (scale_x, scale_y), threshold)
            else:
                raise ValueError(f"Unsupported output_type: {output_type}")

            results.append({"boxes": bboxes, "scores": scores})

        return results

    def _get_rotated_boxes(
        self,
        keys: torch.Tensor,
        label: torch.Tensor,
        score: torch.Tensor,
        scales: Tuple[float, float],
        threshold: float,
    ) -> Tuple[List[List[Tuple[int, int]]], List[float]]:
        """
        Generates rotated rectangular bounding boxes for connected components.

        Args:
            keys (Tensor): Unique instance labels.
            label (Tensor): Label map (H x W).
            score (Tensor): Confidence map (H x W).
            scales (Tuple[float, float]): Scaling factors (x, y) to match original image dimensions.
            threshold (float): Minimum average score for a region to be considered valid.

        Returns:
            Tuple[List[List[int]], List[float]]:
                - List of rotated rectangle bounding boxes as flattened coordinates.
                - List of corresponding confidence scores.
        """
        bounding_boxes = []
        scores = []
        for index in range(1, len(keys)):
            i = keys[index]
            ind = label == i
            ind_np = ind.data.cpu().numpy()
            points = np.array(np.where(ind_np)).transpose((1, 0))
            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue
            score_i = score[ind].mean().item()
            if score_i < threshold:
                label[ind] = 0
                continue

            rect = compute_min_area_rect(points[:, ::-1])
            alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
            rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
            bounding_box = get_box_points(rect) * scales

            bounding_box = bounding_box.astype("int32")
            bounding_boxes.append([tuple(point) for point in bounding_box.tolist()])
            scores.append(score_i)
        return bounding_boxes, scores

    def _get_polygons(
        self,
        keys: torch.Tensor,
        label: torch.Tensor,
        score: torch.Tensor,
        scales: Tuple[float, float],
        threshold: float,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Generates polygonal bounding boxes using OpenCV contours for connected components.

        Note:
            Requires OpenCV backend (`cv2`) to be available.

        Args:
            keys (Tensor): Unique labels.
            label (Tensor): Label map (H x W).
            score (Tensor): Score map (H x W).
            scales (Tuple[float, float]): Scaling factors (x, y).
            threshold (float): Minimum average score for a valid region.

        Returns:
            Tuple[List[List[int]], List[float]]:
                - List of polygon contour bounding boxes as flattened coordinates.
                - List of corresponding confidence scores.
        """
        requires_backends(self, "cv2")
        bounding_boxes = []
        scores = []
        for index in range(1, len(keys)):
            i = keys[index]
            ind = label == i
            ind_np = ind.data.cpu().numpy()
            points = np.array(np.where(ind_np)).transpose((1, 0))
            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue
            score_i = score[ind].mean().item()
            if score_i < threshold:
                label[ind] = 0
                continue

            binary = np.zeros(label.shape, dtype="uint8")
            binary[ind_np] = 1
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_box = contours[0] * scales

            bounding_box = bounding_box.astype("int32")
            bounding_boxes.append(bounding_box.reshape(-1).tolist())
            scores.append(score_i)
        return bounding_boxes, scores


__all__ = ["FastImageProcessor"]
