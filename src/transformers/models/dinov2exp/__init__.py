from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    is_torchvision_available,
)


_import_structure = {"configuration_dinov2exp": ["DINOv2ExpConfig", "DINOv2ExpOnnxConfig"]}

try:
    if not is_torchvision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_dinov2exp_fast"] = ["DINOv2ExpImageProcessorFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_dinov2exp"] = [
        "DINOv2ExpForImageClassification",
        "DINOv2ExpModel",
        "DINOv2ExpPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_dinov2exp"] = [
        "TFDINOv2ExpForImageClassification",
        "TFDINOv2ExpModel",
        "TFDINOv2ExpPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_dinov2exp"] = [
        "FlaxDINOv2ExpForImageClassification",
        "FlaxDINOv2ExpModel",
        "FlaxDINOv2ExpPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_dinov2exp import DINOv2ExpConfig, DINOv2ExpOnnxConfig

    try:
        if not is_torchvision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_dinov2exp_fast import DINOv2ExpImageProcessorFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_dinov2exp import (
            DINOv2ExpForImageClassification,
            DINOv2ExpModel,
            DINOv2ExpPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_dinov2exp import (
            TFDINOv2ExpForImageClassification,
            TFDINOv2ExpModel,
            TFDINOv2ExpPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_dinov2exp import (
            FlaxDINOv2ExpForImageClassification,
            FlaxDINOv2ExpModel,
            FlaxDINOv2ExpPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
