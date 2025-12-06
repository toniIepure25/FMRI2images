#!/usr/bin/env python3
"""
Smoke tests for compare_evals.py and _report_utils.py.

Tests helper functions and full pipeline with mock data.
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def test_report_utils():
    """Test helper functions from _report_utils.py."""
    print("[Test] Report utilities")
    
    # Import utilities from eval module
    from fmri2img.eval._report_utils import (
        load_eval_json,
        guess_run_name,
        bootstrap_ci,
        format_mean_ci
    )
    
    # Test load_eval_json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {"test": "data", "value": 42}
        json.dump(test_data, f)
        temp_json = Path(f.name)
    
    try:
        loaded = load_eval_json(temp_json)
        assert loaded["test"] == "data"
        assert loaded["value"] == 42
    finally:
        temp_json.unlink()
    
    # Test guess_run_name
    path = Path("outputs/reports/subj01/auto_with_adapter/recon_eval.json")
    name = guess_run_name(path)
    assert "adapter" in name.lower() or "auto" in name.lower()
    
    # Test bootstrap_ci
    values = np.array([0.5, 0.6, 0.7, 0.8])
    low, high = bootstrap_ci(values, boots=100, seed=42)
    assert low < np.mean(values) < high
    assert 0 <= low <= 1
    assert 0 <= high <= 1
    
    # Test format_mean_ci
    formatted = format_mean_ci(0.612, 0.571, 0.653)
    assert "0.612" in formatted
    assert "±" in formatted
    
    print("✅ Report utils test passed")


def test_help_output():
    """Test help output is available."""
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "scripts/compare_evals.py", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Help should exit with 0"
    assert "--report-dir" in result.stdout
    assert "--out-csv" in result.stdout
    assert "--out-tex" in result.stdout
    assert "--out-md" in result.stdout
    assert "--out-fig" in result.stdout
    assert "--boots" in result.stdout
    print("✅ Help output test passed")


def test_full_pipeline_mock():
    """Test full pipeline with mock JSON and CSV data."""
    import subprocess
    
    print("[Test] Full pipeline with mock data")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock directory structure
        run1_dir = tmpdir / "run1"
        run2_dir = tmpdir / "run2"
        run1_dir.mkdir()
        run2_dir.mkdir()
        
        # Mock evaluation JSON 1 (no adapter)
        json1_data = {
            "subject": "subj01",
            "clip_space": "512-D (base)",
            "clip_dim": 512,
            "use_adapter": False,
            "encoder": "mlp",
            "n_samples": 10,
            "clipscore": {"mean": 0.612, "std": 0.082},
            "retrieval": {"R@1": 0.487, "R@5": 0.765, "R@10": 0.843},
            "ranking": {"mean_rank": 3.2, "median_rank": 2.0, "mrr": 0.571}
        }
        
        json1_path = run1_dir / "recon_eval.json"
        with open(json1_path, 'w') as f:
            json.dump(json1_data, f)
        
        # Mock per-sample CSV 1
        csv1_data = {
            "nsdId": [12345 + i for i in range(10)],
            "clipscore": np.random.uniform(0.5, 0.7, 10),
            "rank": np.random.randint(1, 20, 10),
            "r@1": np.random.randint(0, 2, 10),
            "r@5": np.random.randint(0, 2, 10),
            "r@10": np.random.randint(0, 2, 10),
        }
        csv1_df = pd.DataFrame(csv1_data)
        csv1_path = run1_dir / "recon_eval.csv"
        csv1_df.to_csv(csv1_path, index=False)
        
        # Mock evaluation JSON 2 (with adapter)
        json2_data = {
            "subject": "subj01",
            "clip_space": "1024-D (target)",
            "clip_dim": 1024,
            "use_adapter": True,
            "encoder": "mlp",
            "model_id": "stabilityai/stable-diffusion-2-1",
            "n_samples": 10,
            "clipscore": {"mean": 0.654, "std": 0.092},
            "retrieval": {"R@1": 0.543, "R@5": 0.812, "R@10": 0.891},
            "ranking": {"mean_rank": 2.1, "median_rank": 1.0, "mrr": 0.612}
        }
        
        json2_path = run2_dir / "recon_eval.json"
        with open(json2_path, 'w') as f:
            json.dump(json2_data, f)
        
        # Mock per-sample CSV 2
        csv2_data = {
            "nsdId": [12345 + i for i in range(10)],
            "clipscore": np.random.uniform(0.6, 0.8, 10),
            "rank": np.random.randint(1, 15, 10),
            "r@1": np.random.randint(0, 2, 10),
            "r@5": np.random.randint(0, 2, 10),
            "r@10": np.random.randint(0, 2, 10),
        }
        csv2_df = pd.DataFrame(csv2_data)
        csv2_path = run2_dir / "recon_eval.csv"
        csv2_df.to_csv(csv2_path, index=False)
        
        # Output paths
        out_csv = tmpdir / "compare.csv"
        out_tex = tmpdir / "compare.tex"
        out_md = tmpdir / "compare.md"
        out_fig = tmpdir / "compare.png"
        
        # Run comparison script
        result = subprocess.run([
            sys.executable,
            "scripts/compare_evals.py",
            "--report-dir", str(tmpdir),
            "--out-csv", str(out_csv),
            "--out-tex", str(out_tex),
            "--out-md", str(out_md),
            "--out-fig", str(out_fig),
            "--boots", "100",  # Faster for testing
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise AssertionError(f"Script failed with exit code {result.returncode}")
        
        # Check outputs exist
        assert out_csv.exists(), "CSV output not created"
        assert out_tex.exists(), "LaTeX output not created"
        assert out_md.exists(), "Markdown output not created"
        assert out_fig.exists(), "Figure output not created"
        
        # Check CSV content
        df = pd.read_csv(out_csv)
        assert len(df) == 2, "Should have 2 runs"
        assert "clipscore_mean" in df.columns
        assert "r1" in df.columns
        assert "clipscore_ci_low" in df.columns
        
        # Check LaTeX content
        tex_content = out_tex.read_text()
        assert "\\begin{table}" in tex_content
        assert "CLIPScore" in tex_content
        assert "R@1" in tex_content
        assert "±" in tex_content
        
        # Check Markdown content
        md_content = out_md.read_text()
        assert "# Reconstruction Evaluation Comparison" in md_content
        assert "Best R@1" in md_content
        assert "Best CLIPScore" in md_content
        assert "adapter" in md_content.lower()
        assert "95% bootstrap" in md_content.lower()
        
        print("✅ Full pipeline test passed")
        print(f"  - Created {len(df)} run comparison")
        print(f"  - CSV: {len(df.columns)} columns")
        print(f"  - LaTeX: {len(tex_content)} chars")
        print(f"  - Markdown: {len(md_content)} chars")
        print(f"  - Figure: {out_fig.stat().st_size} bytes")


def main():
    """Run all smoke tests."""
    print("\n" + "="*80)
    print("  Compare Evals Smoke Tests")
    print("="*80 + "\n")
    
    tests = [
        ("Report Utilities", test_report_utils),
        ("Help Output", test_help_output),
        ("Full Pipeline (Mock Data)", test_full_pipeline_mock),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n[Test] {name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print(f"  Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    if failed > 0:
        print("❌ Some tests failed!")
        return 1
    else:
        print("✅ All smoke tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
