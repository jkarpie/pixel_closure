cd /Users/jkarpie-admin/work/building/pixel_closure
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PIXEL_KERNEL_WORKERS=4 PIXEL_ASSEMBLY_WORKERS=4 PIXEL_POSTERIOR_WORKERS=4 PIXEL_PROGRESS=1
B=campaign/proposal_inflation_2026-09-14
C=campaign/sampler_test_2026-09-14
P=../pixel/venv/bin/python
GRID="1,1.5,2.5,4,6,10,16,25"
fit() { $P -m campaign.prior_scan fit --sampler gaussian --q 2 --suite closure_$1_truth_small --cells $C/cells_$2.json --modes $3 --out $4 --proposal-scales $5; }
( fit JAM c_config "exp both" $B/pilot_JAM $GRID; fit JAM e_config "dy exp" $B/pilot_JAM $GRID ) > $B/stream_JAM.log 2>&1 &
( fit NNPDF c_config "exp" $B/pilot_NNPDF $GRID; fit NNPDF e_config "dy both" $B/pilot_NNPDF $GRID ) > $B/stream_NNPDF.log 2>&1 &
( for s in 1 2 4 8 16; do fit JAM e_config dy $B/fixed_$s $s; done ) > $B/stream_ladder.log 2>&1 &
wait
echo ALL_DONE >> $B/stream_JAM.log
