GPUS=2 GPUS_PER_NODE=2 CPUS_PER_TASK=10 ./tools/slurm_train.sh whitehill upernetswin configs/swin/upernet_swin_joint.py


# GPUS=2 GPUS_PER_NODE=2 CPUS_PER_TASK=10 ./tools/slurm_train.sh whitehill upernetr50joint0 configs/upernet/upernet_r50joint_512x512_80k_ade20k.py
# --resume-from /home/cjmclaughlin/mmseg_jobs/latest.pth