set -x
cd /Users/jkarpie-admin/work/building/pixel_closure
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PIXEL_KERNEL_WORKERS=2 PIXEL_ASSEMBLY_WORKERS=2 PIXEL_POSTERIOR_WORKERS=2
P=../pixel/venv/bin/python
run() { $P -m campaign.contour_profiles run --suite closure_$1_truth_small --mode $2 --cell-json campaign/sampler_test_2026-09-14/cell_json/$3__*.json --out campaign/contour_profiles_2026-09-14 $4; }
run JAM synthetic_z_plumbing c_config
run NNPDF synthetic_z_plumbing e_config
run JAM dy e_config "--cache campaign/sampler_test_2026-09-14/inference_cache/closure_JAM_truth_small__q2__dy__e_config__gaussian_reference.npz"
run NNPDF dy e_config "--cache campaign/sampler_test_2026-09-14/inference_cache/closure_NNPDF_truth_small__q2__dy__e_config__gaussian_reference.npz"
run JAM exp c_config "--cache campaign/sampler_test_2026-09-14/inference_cache/closure_JAM_truth_small__q2__exp__c_config__gaussian_reference.npz"
run NNPDF both e_config "--cache campaign/sampler_test_2026-09-14/inference_cache/closure_NNPDF_truth_small__q2__both__e_config__gaussian_reference.npz"
$P -m campaign.contour_profiles plot --out campaign/contour_profiles_2026-09-14
