#!/bin/bash
#SBATCH -o pixel_closure_%A
#SBATCH --job-name=pixel_closure
#SBATCH -A hadstruc24s
#SBATCH --partition=24s
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks=1


echo "Batch job finished"
export PIXEL_PROGRESS=1 


echo "Calling kernel maker"
PIXEL_PROGRESS=1 ./run_closure.sh --truth JAM --data union --Q 2 \
  --modes both --kernels-only --cores 32 \
  --python ../pixel/venv/bin/python >& out.run_data_kernel_maker 


echo "Calling taskset"
nohup taskset -c 0-63 ../pixel/venv/bin/python \
  -m campaign.run_full_scale_stream_sweep sweep \
  --out campaign/full_scale_core64 \
  --parameterizations q --truths JAM --data union \
  --scales 2 --modes exp --stream-counts 1,2,4,8 \
  --samples 500 > campaign/full_scale_core64.log
echo $!



echo "Batch job finished"
