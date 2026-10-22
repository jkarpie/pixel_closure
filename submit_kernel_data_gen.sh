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

for Q in mc 2 3 
do 
#echo "Calling kernel maker"
PIXEL_PROGRESS=1 ./run_closure.sh --truth JAM --data union --Q $Q \
  --modes both --kernels-only --cores 32 \
  --python ../pixel/venv/bin/python >& out.run_data_kernel_maker 

PIXEL_PROGRESS=1 ./run_closure.sh --truth NNPDF --data union --Q $Q \
  --modes both --kernels-only --cores 32 \
  --python ../pixel/venv/bin/python >& out.run_data_kernel_maker 


done
echo "Batch job finished"
