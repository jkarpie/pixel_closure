cd /Users/jkarpie-admin/work/building/pixel_closure
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PIXEL_KERNEL_WORKERS=4 PIXEL_ASSEMBLY_WORKERS=4 PIXEL_POSTERIOR_WORKERS=4
B=campaign/sampler_exact_width_2026-09-15; C=campaign/sampler_test_2026-09-14; P=../pixel/venv/bin/python
fit() { $P -m campaign.prior_scan fit --sampler gaussian --q 2 --suite closure_$1_truth_small --cells $C/cells_$2.json --modes $3 --out $B; }
( fit JAM c_config "dy exp both synthetic_z_plumbing"; fit JAM e_config "dy exp both" ) > $B/stream_JAM.log 2>&1 &
( sleep 5; fit NNPDF c_config "dis dy exp both synthetic_z_plumbing"; fit NNPDF e_config "dy exp both synthetic_z_plumbing" ) > $B/stream_NNPDF.log 2>&1 &
wait; echo ALL_DONE >> $B/stream_JAM.log
