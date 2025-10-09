import json, os, argparse, fsspec
from tqdm import tqdm
from fmri2img.io.s3 import s3_ls

def main(prefix="s3://natural-scenes-dataset/nsddata", out="manifests/nsd.jsonl", anon=True, limit=None):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Example: list a subset (you'll refine using the Data Manual)
    # Find some stimuli (COCO/NSD selection) and matching betas for subj01:
    stim_keys = s3_ls("natural-scenes-dataset/nsddata_stimuli/**", anon=anon)
    beta_keys = s3_ls("natural-scenes-dataset/nsddata_betas/ppdata/subj01/**", anon=anon)

    # TODO: read the official trial tables from nsddata/ to align (image_id, run, trial)
    # Here we just pair first N for smoke-test:
    pairs = list(zip(sorted(beta_keys), sorted(stim_keys)))
    if limit: pairs = pairs[:limit]

    with open(out, "w") as f:
        for i, (beta, stim) in enumerate(tqdm(pairs, desc="manifest")):
            rec = {
                "trial_id": i,
                "bold": f"s3://{beta}" if not beta.startswith("s3://") else beta,
                "stim": f"s3://{stim}" if not stim.startswith("s3://") else stim,
                "subject": "subj01",
            }
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
