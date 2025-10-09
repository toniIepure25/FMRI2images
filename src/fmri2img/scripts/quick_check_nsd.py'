# quick_check_nsd.py
import fsspec

fs = fsspec.filesystem("s3", anon=True)
print(fs.ls("natural-scenes-dataset"))  # top-level keys
print(fs.glob("natural-scenes-dataset/nsddata_stimuli/**")[:20])  # sample
