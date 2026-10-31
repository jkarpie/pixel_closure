#!/bin/bash
#SBATCH -o pixel_evidence_%A
#SBATCH --job-name=pixel_evidence
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



../pixel/venv/bin/python -m campaign.scan_full_q_normalizations --out campaign/NNLO_RUNS/JAM_Q2_exp_prod_smallprior/normalization_evidence_scan --chain campaign/NNLO_RUNS/JAM_Q2_exp_prod_smallprior/streams_16/q/jam/union/truthQ_2/exp/stream_01/chain.npz > campaign/NNLO_RUNS/JAM_Q2_exp_prod_smallprior/normalization_evidence_scan.log


echo "Batch job finished"
