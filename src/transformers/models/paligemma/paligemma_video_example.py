import torch
import numpy as np
from torchcodec.decoders import VideoDecoder

from transformers import (
    AutoVideoProcessor,
    AutoProcessor,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
)


def main():
    # ============================================================
    # Step 1: Configure PaliGemma with VideoPrism video backbone
    # ============================================================

    # Create PaliGemma config with video support
    # The video_config enables VideoPrism as the video encoder
    config = PaliGemmaConfig(
        video_config={
            "model_type": "videoprism_vision_model",
            "hidden_size": 768,  # VideoPrism base model hidden size
            "num_frames": 16,
            "image_size": 288,
        }
    )

    # Initialize model (for demo purposes - in practice you'd load pretrained weights)
    print("Initializing PaliGemma with VideoPrism video backbone...")
    model = PaliGemmaForConditionalGeneration(config)
    model = model.to(dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print(f"Video tower initialized: {model.video_tower is not None}")
    print(f"Video projector initialized: {model.video_multi_modal_projector is not None}")

    # ============================================================
    # Step 2: Load video processor and text processor
    # ============================================================

    # Use VideoPrism's video processor for video preprocessing
    video_processor = AutoVideoProcessor.from_pretrained("MHRDYN7/videoprism-base-f16r288")

    # Use PaliGemma's processor for text tokenization
    # Note: In practice, you'd use a processor that handles both video and text

    # ============================================================
    # Step 3: Load and process video
    # ============================================================

    video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

    print(f"\nLoading video from: {video_url}")
    vr = VideoDecoder(video_url)
    frame_idx = np.arange(0, 64)  # Sample 64 frames
    video_frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W

    print(f"Video frames shape: {video_frames.shape}")

    # Process video frames
    processed_video = video_processor(video_frames, return_tensors="pt")
    processed_video = {k: v.to(model.device, model.dtype) for k, v in processed_video.items()}

    print(f"Processed video shape: {processed_video['pixel_values_videos'].shape}")

    # ============================================================
    # Step 4: Get video features
    # ============================================================

    print("\nExtracting video features...")
    with torch.no_grad():
        video_features = model.get_video_features(processed_video["pixel_values_videos"])

    print(f"Video features shape: {video_features.shape}")
    # Expected shape: (batch_size, video_length, embed_dim)
    # For VideoPrism base: (1, 4096, projection_dim)

    # ============================================================
    # Step 5: Full forward pass with text prompt
    # ============================================================

    # Prepare text input with image tokens (video uses same tokens)
    prompt = "Describe what is happening in this video."

    # Process text - this adds image tokens for the visual content
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma2-3b-mix-224")
    text_inputs = tokenizer(prompt, return_tensors="pt")
    text_inputs = {k: v.to(model.device) for k, v in text_inputs.items()}

    print(f"\nRunning forward pass with video and text...")
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values_videos=processed_video["pixel_values_videos"],
        )

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Image/Video hidden states shape: {outputs.image_hidden_states.shape}")

    print("\nDone! PaliGemma with video support is working.")


if __name__ == "__main__":
    main()
