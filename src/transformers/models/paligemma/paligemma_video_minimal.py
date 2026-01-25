"""
Minimal example: PaliGemma with VideoPrism video backbone
"""

import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, PaliGemmaConfig, PaliGemmaForConditionalGeneration


# 1. Create PaliGemma config with video support
config = PaliGemmaConfig(
    video_config={"model_type": "videoprism_vision_model"}
)

# 2. Initialize model with video backbone
model = PaliGemmaForConditionalGeneration(config)
model = model.to(dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 3. Load and process video
video_processor = AutoVideoProcessor.from_pretrained("MHRDYN7/videoprism-base-f16r288")

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64)
video_frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W

processed_video = video_processor(video_frames, return_tensors="pt")
processed_video = {k: v.to(model.device, model.dtype) for k, v in processed_video.items()}

# 4. Extract video features
with torch.no_grad():
    video_features = model.get_video_features(processed_video["pixel_values_videos"])

print(f"Video features shape: {video_features.shape}")
# Output: torch.Size([1, 4096, 2048])  # (batch, video_len, projection_dim)
