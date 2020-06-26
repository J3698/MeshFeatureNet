#!/usr/bin/env bash

# This script serves only as a reference for how the OFF files in the original
# ModelNet40 dataset were converted to OBJ files for SoftRasterizer. The working
# directory is recursively searched, and any OFF files are replaced by OBJ files
# if they can be succesfully converted. Program "off2obj" should be installed.

for f in $(find . -name '*.off'); do
    filename="${f%.*}"
    off2obj $f > "${filename}.obj" && rm $f
done

