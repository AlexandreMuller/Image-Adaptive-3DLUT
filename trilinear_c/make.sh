#!/usr/bin/env bash

cd "$(dirname "$0")"
echo "Building trilinear extension via PyTorch..."
python build.py
