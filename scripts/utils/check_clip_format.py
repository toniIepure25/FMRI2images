import pandas as pd
df = pd.read_parquet("outputs/clip_cache/clip.parquet")
print("Columns (first 20):", df.columns.tolist()[:20])
print("Shape:", df.shape)
print("Index name:", df.index.name)
print("Dtypes (first 5):", df.dtypes.tolist()[:5])
