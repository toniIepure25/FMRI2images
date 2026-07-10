#!/bin/bash
cd /tmp/demo_export_v2
for d in cases/nsd_*; do
    name=$(basename "$d")
    tar czf "/tmp/case_${name}.tar.gz" "$d"
    echo "$name: $(du -sh "/tmp/case_${name}.tar.gz" | cut -f1)"
done
