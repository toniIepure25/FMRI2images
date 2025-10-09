import argparse, itertools
from fmri2img.data.nsd_stream import stream_samples

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="manifests/nsd.jsonl")
parser.add_argument("--limit", type=int, default=5)
args = parser.parse_args()

for sample in itertools.islice(stream_samples(args.manifest), args.limit):
    print(sample["trial_id"], sample["subject"], sample["image"].size, sample["bold"].shape)
