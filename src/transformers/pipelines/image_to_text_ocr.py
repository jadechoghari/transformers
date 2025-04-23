from typing import Any, Dict, List, Union
from .base import Pipeline
import torch
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
if is_vision_available():
    from ..image_utils import load_image

def crop_regions_from_bboxes(images, all_boxes):
    cropped_regions = []
    if images.ndim == 3:
        images = images.unsqueeze(0)
        all_boxes = [all_boxes]
    for img, item in zip(images, all_boxes):
        for poly in item["boxes"]:
            xs, ys = zip(*poly)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cropped = img[:, y1:y2, x1:x2]
            cropped_regions.append(cropped)
    return cropped_regions

class OCRPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model and self.image_process will be parsed and taken from fast by default
        self.text_recognition_model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')
        self.recog_processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
        
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, image):
        # image can be either url or PIL object
        image = load_image(image) # return PIL object regardless
        target_size = torch.IntTensor([[image.height, image.width]])
        inputs = self.image_processor(image, return_tensors="pt") # return image in pixel space
        
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        # Step 1: Run FAST model
        target_size = model_inputs.pop("target_size")
        detection_outputs = self.model(**model_inputs)

        # Step 2: Postprocess detection results
        all_boxes = self.image_processor.post_process_text_detection(detection_outputs, target_sizes=target_size, threshold=0.88)
        # Step 3: Crop regions from each image using the detected boxes
        cropped_regions = crop_regions_from_bboxes(model_inputs['pixel_values'], all_boxes)

        from torchvision.transforms.functional import to_pil_image

        # Use the same mean/std as your image processor (usually ImageNet stats)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        def denormalize(crop, mean, std):
            for t, m, s in zip(crop, mean, std):
                t.mul_(s).add_(m)
            return crop.clamp(0, 1)

        pil_crops = []
        for crop in cropped_regions:
            crop = denormalize(crop.clone(), mean, std)  # clone to avoid inplace ops
            pil = to_pil_image(crop)
            pil_crops.append(pil)
        # Step 4: Forward pass through TrOCR or MGP-STR on cropped regions
        recog_inputs = self.recog_processor(images=pil_crops, return_tensors="pt")
        return recog_inputs


    def postprocess(self, model_outputs):
        recog_outputs = self.text_recognition_model(**model_outputs)
        generated_text = self.recog_processor.batch_decode(recog_outputs.logits)['generated_text']
        return generated_text
