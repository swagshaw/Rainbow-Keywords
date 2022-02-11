#/bin/bash

# CIL CONFIG
MODE="rm" # joint, gdumb, icarl, rm, ewc, rwalk, bic
# "default": If you want to use the default memory management method.
MEM_MANAGE="default" # default, random, reservoir, uncertainty, prototype.
RND_SEED=1
DATASET="gsc" # gsc
STREAM="offline" # offline, online
EXP="disjoint" # disjoint, blurry10, blurry30
MEM_SIZE=500 # k={200, 500, 1000}
TRANS="" # multiple choices: cutmix, cutout, randaug, autoaug

N_WORKER=8
JOINT_ACC=94.0 # training all the tasks at once.
# FINISH CIL CONFIG ####################

UNCERT_METRIC="vr"
PRETRAIN="" INIT_MODEL="" INIT_OPT="--init_opt"

# iCaRL
FEAT_SIZE=256

# BiC
distilling="" # Normal BiC. If you do not want to use distilling loss, then "".

python main.py --mode bic --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET \
--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED \
--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--n_worker $N_WORKER --n_epoch $N_EPOCH \
--memory_size $MEM_SIZE --transform $TRANS --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC

