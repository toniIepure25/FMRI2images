#!/usr/bin/env python3
"""
Build Text-CLIP Cache for Multi-Task Semantics
==============================================

Generate captions for NSD images and encode them with CLIP text encoder.
This enables multi-task learning: fMRI → image-CLIP + text-CLIP.

Scientific Rationale:
- Text-CLIP captures semantic/linguistic concepts
- Image-CLIP captures visual features
- Joint supervision improves semantic understanding
- Novel for fMRI decoding (not in MindEye2, Brain-Diffuser)

Workflow:
1. Load NSD images from cache/stimuli
2. Generate 1-3 captions per image using BLIP-2 or similar
3. Encode captions with CLIP text encoder (same model as image encoder)
4. Average multiple captions per image (or keep best)
5. Save to cache/clip_embeddings/text_clip.parquet

Usage:
    # Basic usage with BLIP-2
    python scripts/build_text_clip_cache.py \\
        --image-dir cache/stimuli \\
        --output cache/clip_embeddings/text_clip.parquet
    
    # With specific CLIP model
    python scripts/build_text_clip_cache.py \\
        --image-dir cache/stimuli \\
        --clip-model ViT-B-32 \\
        --clip-pretrained openai \\
        --num-captions 3 \\
        --output cache/clip_embeddings/text_clip.parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_captioning_model(model_name: str = "blip2", device: str = "cuda", local_path: str = None):
    """
    Load image captioning model.
    
    Args:
        model_name: "blip2", "blip", or "git"
        device: Device for model
        local_path: Optional local path to model (e.g., ~/models/blip2-opt-2.7b)
    
    Returns:
        (model, processor) tuple
    """
    import torch
    from pathlib import Path
    
    logger.info(f"Loading captioning model: {model_name}")
    
    # Expand local path if provided
    if local_path:
        local_path = Path(local_path).expanduser()
        if not local_path.exists():
            logger.warning(f"Local path {local_path} not found, falling back to HuggingFace download")
            local_path = None
        else:
            logger.info(f"Using local model from {local_path}")
    
    try:
        if model_name == "blip2":
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            # BLIP-2 OPT-2.7B (LARGE: ~15GB download, best quality but VERY slow to download)
            # Consider using "blip" instead for faster setup
            model_id = local_path if local_path else "Salesforce/blip2-opt-2.7b"
            
            if not local_path:
                logger.warning("⚠️  BLIP-2 is a LARGE model (~15GB). Download may take 30+ minutes.")
                logger.warning("⚠️  Consider using --model blip for faster setup (~1GB, good quality)")
                logger.warning("⚠️  Or download manually and use --caption-model-path ~/models/blip2-opt-2.7b")
            
            processor = Blip2Processor.from_pretrained(model_id, local_files_only=bool(local_path))
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                resume_download=not bool(local_path),
                local_files_only=bool(local_path)
            )
            logger.info(f"Loaded BLIP-2 model from {model_id}")
            
        elif model_name == "blip":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            # Original BLIP (faster but lower quality)
            model_id = "Salesforce/blip-image-captioning-large"
            processor = BlipProcessor.from_pretrained(model_id)
            model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
            logger.info(f"Loaded BLIP model from {model_id}")
            
        elif model_name == "git":
            from transformers import AutoProcessor, AutoModelForCausalLM
            # GIT (Microsoft, good quality)
            model_id = "microsoft/git-large-coco"
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            logger.info(f"Loaded GIT model from {model_id}")
            
        else:
            raise ValueError(f"Unknown captioning model: {model_name}")
        
        model.eval()
        return model, processor
        
    except ImportError as e:
        logger.error("transformers not installed: pip install transformers")
        raise e


def load_clip_text_encoder(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda"
):
    """
    Load CLIP text encoder (must match image encoder used for image-CLIP cache).
    
    Args:
        model_name: CLIP architecture (e.g., "ViT-B-32", "ViT-L-14")
        pretrained: Pretrained weights (e.g., "openai", "laion2b_s34b_b79k")
        device: Device for model
    
    Returns:
        (model, tokenizer) tuple
    """
    logger.info(f"Loading CLIP text encoder: {model_name} ({pretrained})")
    
    try:
        import open_clip
        import torch
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        
        model.eval()
        logger.info(f"Loaded CLIP text encoder: {model_name}")
        
        return model, tokenizer
        
    except ImportError as e:
        logger.error("open_clip_torch not installed: pip install open-clip-torch")
        raise e


def generate_captions(
    image_path: Path,
    caption_model,
    caption_processor,
    num_captions: int = 3,
    device: str = "cuda"
) -> List[str]:
    """
    Generate multiple captions for an image.
    
    Args:
        image_path: Path to image
        caption_model: Captioning model
        caption_processor: Captioning processor
        num_captions: Number of captions to generate
        device: Device
    
    Returns:
        List of caption strings
    """
    from PIL import Image
    import torch
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load {image_path}: {e}")
        return []
    
    captions = []
    
    with torch.no_grad():
        for _ in range(num_captions):
            # Process image
            inputs = caption_processor(images=image, return_tensors="pt").to(device)
            
            # Generate caption with sampling (different each time)
            generated_ids = caption_model.generate(
                **inputs,
                max_length=50,
                num_beams=3,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Decode
            caption = caption_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            captions.append(caption)
    
    return captions


def encode_text_with_clip(
    texts: List[str],
    clip_model,
    clip_tokenizer,
    device: str = "cuda"
) -> np.ndarray:
    """
    Encode text with CLIP text encoder.
    
    Args:
        texts: List of text strings
        clip_model: CLIP model
        clip_tokenizer: CLIP tokenizer
        device: Device
    
    Returns:
        Text embeddings (N, D), L2-normalized
    """
    import torch
    
    # Tokenize
    text_tokens = clip_tokenizer(texts).to(device)
    
    # Encode
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        # L2 normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Build text-CLIP cache for multi-task learning"
    )
    
    # Input/output
    parser.add_argument("--image-dir", default="cache/stimuli",
                       help="Directory with NSD images")
    parser.add_argument("--index-dir", default="data/indices/nsd_index",
                       help="NSD index directory")
    parser.add_argument("--output", default="cache/clip_embeddings/text_clip.parquet",
                       help="Output parquet file")
    
    # Captioning model
    parser.add_argument("--caption-model", choices=["blip2", "blip", "git"],
                       default="blip", help="Image captioning model (default: blip for speed)")
    parser.add_argument("--caption-model-path", type=str, default=None,
                       help="Local path to caption model (e.g., ~/models/blip2-opt-2.7b)")
    parser.add_argument("--num-captions", type=int, default=1,
                       help="Number of captions per image (default: 1 for speed)")
    
    # CLIP model (must match image-CLIP cache)
    parser.add_argument("--clip-model", default="ViT-B-32",
                       help="CLIP model architecture")
    parser.add_argument("--clip-pretrained", default="openai",
                       help="CLIP pretrained weights")
    
    # Processing
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for CLIP encoding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Options
    parser.add_argument("--aggregation", choices=["mean", "max", "first"],
                       default="mean",
                       help="How to aggregate multiple captions")
    parser.add_argument("--limit", type=int, help="Limit number of images (for testing)")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Building Text-CLIP Cache")
    logger.info("=" * 70)
    
    # Load models
    logger.info("\n1. Loading models...")
    caption_model, caption_processor = load_captioning_model(
        args.caption_model, args.device, args.caption_model_path
    )
    clip_model, clip_tokenizer = load_clip_text_encoder(
        args.clip_model, args.clip_pretrained, args.device
    )
    
    # Find all images
    logger.info("\n2. Finding images...")
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Load NSD stimulus info to map nsdId to COCO filenames
    stim_info_path = Path("cache/nsd_stim_info_merged.csv")
    nsd_to_filename = {}
    filename_to_nsd = {}
    
    if stim_info_path.exists():
        logger.info(f"Loading stimulus info from {stim_info_path}")
        stim_info = pd.read_csv(stim_info_path)
        for _, row in stim_info.iterrows():
            nsd_id = int(row['nsdId'])
            coco_id = int(row['cocoId'])
            coco_split = row['cocoSplit']
            filename = f"{coco_id}_{coco_split}.jpg"
            nsd_to_filename[nsd_id] = filename
            filename_to_nsd[filename] = nsd_id
        logger.info(f"Loaded {len(nsd_to_filename)} NSD→filename mappings")
    else:
        logger.warning(f"Stimulus info file not found at {stim_info_path}")
        logger.warning("Will attempt to parse nsdId from filenames directly")
    
    # Get list of image files
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} images in {image_dir}")
    
    if args.limit:
        if filename_to_nsd:
            # Get first N valid NSD images
            valid_files = [f for f in image_files if f.name in filename_to_nsd]
            image_files = valid_files[:args.limit]
        else:
            image_files = image_files[:args.limit]
        logger.info(f"Limited to {len(image_files)} images for testing")
    
    # Process images
    logger.info(f"\n3. Generating captions and encoding with CLIP...")
    logger.info(f"   Captions per image: {args.num_captions}")
    logger.info(f"   Aggregation: {args.aggregation}")
    
    results = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # Extract nsd_id from filename mapping or direct parsing
        nsd_id = None
        
        if filename_to_nsd and image_path.name in filename_to_nsd:
            nsd_id = filename_to_nsd[image_path.name]
        else:
            # Try to parse from filename (e.g., "nsd73000.png" -> 73000)
            try:
                nsd_id = int(image_path.stem.replace("nsd", ""))
            except:
                logger.warning(f"Could not parse nsd_id from {image_path.name}, skipping")
                continue
        
        if nsd_id is None:
            continue
        
        # Generate captions
        captions = generate_captions(
            image_path,
            caption_model,
            caption_processor,
            num_captions=args.num_captions,
            device=args.device
        )
        
        if not captions:
            logger.warning(f"No captions generated for {image_path.name}, skipping")
            continue
        
        # Encode captions with CLIP
        text_embeddings = encode_text_with_clip(
            captions,
            clip_model,
            clip_tokenizer,
            device=args.device
        )  # (N, D)
        
        # Aggregate multiple captions
        if args.aggregation == "mean":
            text_embedding = text_embeddings.mean(axis=0)
        elif args.aggregation == "max":
            # Max pooling across captions
            text_embedding = text_embeddings.max(axis=0)
        elif args.aggregation == "first":
            text_embedding = text_embeddings[0]
        
        # L2 normalize final embedding
        text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
        
        results.append({
            "nsd_id": nsd_id,
            "text_clip_embedding": text_embedding,
            "captions": captions  # Store for inspection
        })
    
    # Create DataFrame
    logger.info(f"\n4. Saving to {args.output}...")
    df = pd.DataFrame(results)
    
    # Save to parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"✅ Saved text-CLIP cache: {len(df)} images")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Embedding dim: {df['text_clip_embedding'].iloc[0].shape[0]}")
    
    # Show example
    logger.info("\n5. Example captions:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        logger.info(f"\n   nsd_id={row['nsd_id']}:")
        for j, caption in enumerate(row['captions'], 1):
            logger.info(f"     {j}. {caption}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Text-CLIP cache built successfully!")
    logger.info("=" * 70)
    logger.info("\nNext step: Use in multi-task training")
    logger.info(f"  --text-clip-cache {args.output}")
    logger.info(f"  --multi-task-enabled")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
