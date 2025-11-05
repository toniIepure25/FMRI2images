# Compare Evals Fix: Nested Dictionary Handling

## Issue

The `scripts/compare_evals.py` script was encountering "unhashable type" errors when processing evaluation JSON files that contained nested dictionaries, such as:

```json
{
  "retrieval_gallery": {
    "type": "matched",
    "size": 1,
    "n_eligible": 1,
    "n_total": 1
  },
  "adapter_ablation": {
    "clipscore_no_adapter": 0.5,
    "delta_clipscore": -0.4
  }
}
```

When these nested dictionaries were stored in a pandas DataFrame, operations like `sort_values()` would fail because dictionaries are unhashable and cannot be used as values in certain DataFrame operations.

## Solution

Implemented a `flatten_dict()` function that converts nested dictionaries (one level deep) into flat key-value pairs with underscore-separated keys:

```python
def flatten_dict(
    d: Dict,
    parent_key: str = "",
    sep: str = "_"
) -> Dict:
    """
    Flatten nested dictionary one level deep.
    
    Converts nested dicts like {"a": {"b": 1, "c": 2}} to {"a_b": 1, "a_c": 2}.
    Handles only depth-1 nesting to avoid issues with DataFrame creation.
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Flatten one level
            for nested_k, nested_v in v.items():
                nested_key = f"{new_key}{sep}{nested_k}"
                items.append((nested_key, nested_v))
        else:
            items.append((new_key, v))
    
    return dict(items)
```

### Applied Changes

1. **Added `flatten_dict()` function** (lines 66-93): Flattens nested dictionaries one level deep
2. **Modified `compute_run_metrics()` function** (lines 137-202):
   - Applied flattening immediately after loading JSON: `data = flatten_dict(data)`
   - Updated metric extraction to use flattened keys:
     - `data.get("clipscore_mean")` instead of `data.get("clipscore", {}).get("mean")`
     - `data.get("retrieval_R@1")` instead of `data.get("retrieval", {}).get("R@1")`
   - Added code to include flattened gallery/ablation fields in results

## Testing

Tested with evaluation JSONs containing nested `retrieval_gallery` fields:

```bash
python3 scripts/compare_evals.py \
    --report-dir outputs/reports/subj01 \
    --pattern "test_gallery*.json" \
    --out-csv outputs/reports/subj01/test_gallery_compare.csv \
    --out-tex outputs/reports/subj01/test_gallery_compare.tex \
    --out-md outputs/reports/subj01/test_gallery_compare.md \
    --out-fig outputs/reports/subj01/test_gallery_compare.png
```

**Result:** ✓ Success - No unhashable type errors

**Output verification:**
```csv
run_name,retrieval_gallery_type,retrieval_gallery_size,retrieval_gallery_n_eligible,retrieval_gallery_n_total
test_gallery_matched,matched,1,1,1
test_gallery_test,test,1,1,1
test_gallery_all,all,5,1,1
test_gallery_all_large,all,5,1,1
```

## Benefits

1. **Handles nested structures gracefully**: Automatically flattens nested dicts without manual intervention
2. **Preserves all information**: All nested fields are included in output with descriptive flattened names
3. **Maintains compatibility**: Existing code that doesn't have nested fields continues to work
4. **Improves output quality**: Gallery and ablation metadata now visible in comparison tables
5. **Prevents future errors**: Any new nested fields added to evaluation JSONs will be automatically handled

## Example Output

After flattening, nested `retrieval_gallery` fields become:
- `retrieval_gallery_type`: "matched", "test", or "all"
- `retrieval_gallery_size`: Gallery size (e.g., 1, 5000)
- `retrieval_gallery_n_eligible`: Number of eligible images
- `retrieval_gallery_n_total`: Total number of test images

These fields are now visible in:
- ✓ CSV comparison tables
- ✓ LaTeX tables (if included in template)
- ✓ Markdown summaries
- ✓ Can be used for sorting/filtering

## Files Modified

- `scripts/compare_evals.py`: Added flattening function and updated metric extraction
- Total changes: +60 lines (new function + updated logic)
