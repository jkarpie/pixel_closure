#!/bin/bash

run=$1

git add -f campaign/${run}/manifest.json
find campaign/${run} -type f \
  \( -name REPORT.json -o -name stream.json -o -name inference_summary.json \) \
  -print0 | xargs -0 git add -f --
