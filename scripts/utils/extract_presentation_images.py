"""Extract specific NSD stimulus images for the thesis defense presentation.

Run this on the JupyterHub pod where NSD data is available:
    python scripts/utils/extract_presentation_images.py

Outputs 4 images to presentation/images/ for the retrieval gallery slide.
"""

import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

NSD_HDF5 = os.environ.get(
    "NSD_HDF5",
    "/home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
)

TARGET_NSD_IDS = [8262, 21279, 55649, 8509]
LABELS = ["surfer", "skier", "seagulls", "fruit_bowl"]

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "presentation" / "images"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading NSD stimuli from: {NSD_HDF5}")
    with h5py.File(NSD_HDF5, "r") as f:
        stimuli = f["imgBrick"]
        for nsd_id, label in zip(TARGET_NSD_IDS, LABELS):
            img_array = stimuli[nsd_id]
            img = Image.fromarray(img_array.astype(np.uint8))
            img_resized = img.resize((256, 256), Image.LANCZOS)
            out_path = OUTPUT_DIR / f"nsd_{nsd_id}_{label}.jpg"
            img_resized.save(out_path, quality=92)
            print(f"  Saved: {out_path.name} ({img.size[0]}x{img.size[1]} -> 256x256)")

    print(f"\nDone. {len(TARGET_NSD_IDS)} images saved to {OUTPUT_DIR}")
    print("Copy presentation/images/ to your local machine for the defense.")


if __name__ == "__main__":
    main()
