#!/bin/bash
echo "Install dependencies"
pip install -r requirements.txt --default-timeout=100
sed -i 's/python = "^3.8, <3.11"/python = "^3.8, <3.12"/' langsam/pyproject.toml
cd langsam
pip install -e .
cd ..

echo "Done"