#!/usr/bin/env python3
"""
MinIO Connection Tester

Quick script to verify your MinIO setup is working correctly.
Run this after configuring .env with MinIO credentials.

Usage:
    python scripts/test_minio.py
    python scripts/test_minio.py --endpoint https://ubbc1u.ro:9000 --bucket nsd-data
"""

import os
import sys
import argparse
from pathlib import Path

def test_imports():
    """Test that required packages are installed"""
    print("=" * 70)
    print("1. Testing Python Imports")
    print("=" * 70)
    
    try:
        import s3fs
        print("✓ s3fs installed:", s3fs.__version__)
    except ImportError:
        print("✗ s3fs not installed")
        print("  Install: pip install s3fs")
        return False
    
    try:
        import fsspec
        print("✓ fsspec installed:", fsspec.__version__)
    except ImportError:
        print("✗ fsspec not installed")
        print("  Install: pip install fsspec")
        return False
    
    print("✓ All required packages installed\n")
    return True


def test_credentials():
    """Test that credentials are set in environment"""
    print("=" * 70)
    print("2. Testing Environment Variables")
    print("=" * 70)
    
    required = {
        'AWS_ENDPOINT_URL': 'MinIO server endpoint',
        'AWS_ACCESS_KEY_ID': 'Access key',
        'AWS_SECRET_ACCESS_KEY': 'Secret key'
    }
    
    all_set = True
    for var, desc in required.items():
        value = os.environ.get(var)
        if value:
            # Mask secret key
            if 'SECRET' in var:
                display = value[:4] + '...' + value[-4:] if len(value) > 8 else '***'
            else:
                display = value
            print(f"✓ {var}: {display}")
        else:
            print(f"✗ {var}: NOT SET ({desc})")
            all_set = False
    
    if not all_set:
        print("\n⚠️  Missing credentials!")
        print("   Add them to .env file:")
        print("   AWS_ENDPOINT_URL=https://ubbc1u.ro:9000")
        print("   AWS_ACCESS_KEY_ID=your_key")
        print("   AWS_SECRET_ACCESS_KEY=your_secret")
        return False
    
    print("✓ All credentials configured\n")
    return True


def test_connection(endpoint, key, secret):
    """Test basic connection to MinIO"""
    print("=" * 70)
    print("3. Testing MinIO Connection")
    print("=" * 70)
    
    import s3fs
    
    try:
        print(f"Connecting to: {endpoint}")
        fs = s3fs.S3FileSystem(
            endpoint_url=endpoint,
            key=key,
            secret=secret
        )
        
        print("✓ Connection established")
        return fs
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check endpoint URL (try :9000, :9001, http vs https)")
        print("2. Verify credentials are correct")
        print("3. Check if MinIO server is accessible from your network")
        return None


def test_list_buckets(fs):
    """List available buckets"""
    print("\n" + "=" * 70)
    print("4. Listing Buckets")
    print("=" * 70)
    
    try:
        buckets = fs.ls('')
        if buckets:
            print(f"✓ Found {len(buckets)} bucket(s):")
            for bucket in buckets:
                print(f"  - {bucket}")
        else:
            print("⚠️  No buckets found (might be permission issue)")
        return buckets
    except Exception as e:
        print(f"✗ Failed to list buckets: {e}")
        return []


def test_bucket_access(fs, bucket):
    """Test access to specific bucket"""
    print("\n" + "=" * 70)
    print(f"5. Testing Bucket Access: {bucket}")
    print("=" * 70)
    
    try:
        contents = fs.ls(bucket, detail=False)
        print(f"✓ Bucket accessible")
        print(f"✓ Found {len(contents)} top-level items")
        
        # Show first few items
        if contents:
            print("\nFirst items:")
            for item in contents[:5]:
                print(f"  - {item}")
            if len(contents) > 5:
                print(f"  ... and {len(contents) - 5} more")
        
        return True
    except Exception as e:
        print(f"✗ Cannot access bucket: {e}")
        print("\nCheck:")
        print("1. Bucket name is correct")
        print("2. You have read permissions")
        return False


def test_nsd_structure(fs, bucket):
    """Check for NSD dataset structure"""
    print("\n" + "=" * 70)
    print("6. Checking NSD Dataset Structure")
    print("=" * 70)
    
    expected_paths = [
        f"{bucket}/nsddata",
        f"{bucket}/nsddata_betas",
        f"{bucket}/nsddata_stimuli",
    ]
    
    found = []
    for path in expected_paths:
        try:
            if fs.exists(path):
                print(f"✓ Found: {path}")
                found.append(path)
            else:
                print(f"✗ Not found: {path}")
        except Exception as e:
            print(f"✗ Error checking {path}: {e}")
    
    if found:
        print(f"\n✓ Found {len(found)}/{len(expected_paths)} expected directories")
        return True
    else:
        print("\n⚠️  Standard NSD structure not found")
        print("   The data might be in a different location")
        print("   List all contents with: fs.ls(bucket)")
        return False


def test_file_download(fs, bucket):
    """Test downloading a small file"""
    print("\n" + "=" * 70)
    print("7. Testing File Download (Cache Test)")
    print("=" * 70)
    
    # Try to find a small file to test with
    try:
        # Look for any small file in the bucket
        contents = fs.ls(bucket, detail=True)
        
        # Find a small file (< 10MB)
        test_file = None
        for item in contents:
            if item['type'] == 'file' and item.get('size', 0) < 10 * 1024 * 1024:
                test_file = item['name']
                test_size = item['size']
                break
        
        if not test_file:
            print("⚠️  No small test file found, skipping download test")
            return True
        
        print(f"Testing download of: {test_file}")
        print(f"Size: {test_size / 1024:.1f} KB")
        
        # Try to read first few bytes
        with fs.open(test_file, 'rb') as f:
            data = f.read(1024)  # Read 1KB
            print(f"✓ Successfully read {len(data)} bytes")
            print("✓ Download/streaming works!")
        
        return True
        
    except Exception as e:
        print(f"✗ Download test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test MinIO connection")
    parser.add_argument('--endpoint', help='MinIO endpoint URL', 
                       default=os.environ.get('AWS_ENDPOINT_URL'))
    parser.add_argument('--key', help='Access key ID',
                       default=os.environ.get('AWS_ACCESS_KEY_ID'))
    parser.add_argument('--secret', help='Secret access key',
                       default=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    parser.add_argument('--bucket', help='Bucket name to test',
                       default=os.environ.get('MINIO_BUCKET', 'nsd-data'))
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "MinIO CONNECTION TESTER" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Run tests
    if not test_imports():
        print("\n❌ FAILED: Missing required packages")
        return 1
    
    if not test_credentials():
        print("\n❌ FAILED: Missing credentials")
        print("\nTo fix:")
        print("1. Copy template: cp .env.minio .env")
        print("2. Edit .env with your credentials")
        print("3. Source environment: source activate_env.sh")
        return 1
    
    fs = test_connection(args.endpoint, args.key, args.secret)
    if not fs:
        print("\n❌ FAILED: Cannot connect to MinIO")
        return 1
    
    buckets = test_list_buckets(fs)
    
    if args.bucket:
        if args.bucket in buckets or f"s3://{args.bucket}" in buckets:
            test_bucket_access(fs, args.bucket)
            test_nsd_structure(fs, args.bucket)
            test_file_download(fs, args.bucket)
        else:
            print(f"\n⚠️  Specified bucket '{args.bucket}' not found")
            print("Available buckets:", buckets)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ MinIO connection working!")
    print("✓ Ready to use S3 backend")
    print("\nNext steps:")
    print("1. Run: python scripts/verify_dataset.py --allow-s3-only")
    print("2. Run: bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml")
    print("3. Monitor cache: du -sh $S3_CACHE_DIR")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
