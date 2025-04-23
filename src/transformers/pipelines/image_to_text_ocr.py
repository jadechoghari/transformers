from typing import Any, Dict, List, Union
from .base import Pipeline
import torch
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
if is_vision_available():
    from ..image_utils import load_image
from PIL import Image

def crop_regions_from_bboxes(images, all_boxes):
    """
    images: list of PIL.Image (original images)
    all_boxes: list of dicts, one per image (each with key "boxes" as list of polygons)
    Returns: list of lists of PIL.Image (crops per image)
    """
    # handle single image input for convenience
    if isinstance(images, Image.Image):
        images = [images]
    cropped_batch = []
    for img, boxes_dict in zip(images, all_boxes):
        crops = []
        for poly in boxes_dict["boxes"]:
            xs, ys = zip(*poly)
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            crop = img.crop((x1, y1, x2, y2))
            crops.append(crop)
        cropped_batch.append(crops)
    return cropped_batch




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
        self.image = image
        target_size = torch.IntTensor([[image.height, image.width]])
        inputs = self.image_processor(image, return_tensors="pt") # return image in pixel space
        
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        # Step 1: Run FAST model
        target_size = model_inputs.pop("target_size")
        detection_outputs = self.model(**model_inputs)

        # Step 2: Postprocess detection results
        all_boxes = self.image_processor.post_process_text_detection(detection_outputs, target_sizes=target_size, threshold=0.7,  output_type="boxes")

        # Step 3: Crop the region needed
        cropped_regions = crop_regions_from_bboxes(self.image, all_boxes)

        # Flatten if needed
        if cropped_regions and isinstance(cropped_regions[0], list):
            cropped_regions = [item for sublist in cropped_regions for item in sublist]

        # Step 4: Forward pass through TrOCR or MGP-STR on cropped regions
        recog_inputs = self.recog_processor(images=cropped_regions, return_tensors="pt")
        return recog_inputs


    def postprocess(self, model_outputs):
        recog_outputs = self.text_recognition_model(**model_outputs)
        generated_text = self.recog_processor.batch_decode(recog_outputs.logits)['generated_text']
        return generated_text
