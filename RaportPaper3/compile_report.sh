#!/bin/bash
# Quick script to compile the LaTeX report
# Usage: ./compile_report.sh

set -e  # Exit on error

echo "========================================="
echo "Compiling LaTeX Experimental Report"
echo "========================================="
echo ""

# Check if we're in the report directory
if [ ! -f "experimental_report.tex" ]; then
    echo "Error: experimental_report.tex not found!"
    echo "Please run this script from the report/ directory"
    exit 1
fi

# Check for LaTeX installation
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found!"
    echo "Please install a LaTeX distribution (TeX Live, MiKTeX, or MacTeX)"
    exit 1
fi

# Check for latexmk (optional but recommended)
if command -v latexmk &> /dev/null; then
    echo "Using latexmk for compilation..."
    latexmk -pdf -interaction=nonstopmode experimental_report.tex
else
    echo "latexmk not found, using manual compilation..."
    echo "Step 1/4: First pdflatex pass..."
    pdflatex -interaction=nonstopmode experimental_report.tex
    
    echo "Step 2/4: Running bibtex..."
    bibtex experimental_report
    
    echo "Step 3/4: Second pdflatex pass..."
    pdflatex -interaction=nonstopmode experimental_report.tex
    
    echo "Step 4/4: Final pdflatex pass..."
    pdflatex -interaction=nonstopmode experimental_report.tex
fi

echo ""
echo "========================================="
echo "Compilation complete!"
echo "========================================="
echo ""
echo "Output file: experimental_report.pdf"
echo ""

# Check if PDF was created
if [ -f "experimental_report.pdf" ]; then
    echo "✓ PDF successfully generated"
    
    # Get file size
    size=$(du -h experimental_report.pdf | cut -f1)
    echo "  Size: $size"
    
    # Count pages (if pdfinfo is available)
    if command -v pdfinfo &> /dev/null; then
        pages=$(pdfinfo experimental_report.pdf | grep Pages | awk '{print $2}')
        echo "  Pages: $pages"
    fi
    
    echo ""
    echo "To view the PDF:"
    echo "  Linux:  xdg-open experimental_report.pdf"
    echo "  macOS:  open experimental_report.pdf"
    echo "  Or use: make view"
else
    echo "✗ Error: PDF was not generated"
    echo "  Check the .log file for errors"
    exit 1
fi

echo ""
echo "To clean auxiliary files: make clean"
echo "To clean all files:       make cleanall"
echo ""
