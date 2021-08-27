#!/bin/sh
for filename in /scratch/*.ipynb; do
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $filename
done