#!/bin/bash
#SBATCH -o pixel_closure_%A
#SBATCH --job-name=pixel_closure
#SBATCH -A hadstruc24s
#SBATCH --partition=24s
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks=1


dataset=exp
truth=JAM
Q=2

echo "Batch job finished"
export PIXEL_PROGRESS=1 


echo "Calling kernel maker"
PIXEL_PROGRESS=1 ./run_closure.sh --truth $truth --data union --Q $Q \
  --modes $dataset --kernels-only --cores 32 \
  --python ../pixel/venv/bin/python >&  campaign/${truth}_Q${Q}_${dataset}_prod.kern_data_gen.log 

rm -rf campaign/${truth}_Q${Q}_${dataset}_prod

./run_closure.sh \
  --truth $truth --data union --Q $Q --modes $dataset \
  --parameterization q --warmup 1000 --samples 25000 --streams 16 \
  --out campaign/${truth}_Q${Q}_${dataset}_prod --no-plots \
  --python ../pixel/venv/bin/python \
  > campaign/${truth}_Q${Q}_${dataset}_prod.log 


echo "Batch job finished"
