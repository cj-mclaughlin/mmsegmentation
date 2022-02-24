GPUS=2 GPUS_PER_NODE=2 CPUS_PER_TASK=10 ./tools/slurm_train.sh whitehill upernetr50ds configs/upernet/upernet_r50deepsup_512x512_80k_ade20k.py --work-dir /home/cjmclaughlin/mmseg_jobs/

# --resume-from /home/cjmclaughlin/mmseg_jobs/latest.pth