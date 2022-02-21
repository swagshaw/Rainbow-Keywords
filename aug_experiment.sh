#/bin/bash

# CIL CONFIG
MODE="rainbow_keywords" # finetune ,native_rehearsal,joint, rwalk, icarl, rainbow_keywords, ewc, bic, gdumb
# "default": If you want to use the default memory management method.
MEM_MANAGE="default" # default, random, reservoir, uncertainty, prototype.
RND_SEED=1
DATASET="gsc" # gsc
STREAM="online" # offline, online
EXP="disjoint" # disjoint, blurry10, blurry30
MEM_SIZE=500 # k={200, 500, 1000}
TRANS="" # multiple choices:"Additional train transforms [mixup, specmix, specaugment]"
TOPK=1
N_WORKER=5
JOINT_ACC=0.94 # training all the tasks at once.
N_TASKS=6
N_INIT_CLS=15
N_CLS_A_TASK=3
MODEL_NAME="tcresnet8"
OPT_NAME="adam"
SCHED_NAME="cos"
LR=0.1
BATCHSIZE=128
N_EPOCH=50
# FINISH CIL CONFIG ####################

UNCERT_METRIC="vr"
PRETRAIN="" INIT_MODEL="" INIT_OPT="--init_opt"

# for iCaRL
FEAT_SIZE=256

# for BiC
distilling="" # Normal BiC. If you do not want to use distilling loss, then "".

python main.py --mode efficient_memory --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET \
--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED \
--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--n_worker $N_WORKER --n_epoch $N_EPOCH \
--memory_size 300 --transform "kd_trick" --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC

python main.py --mode efficient_memory --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET \
--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED \
--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--n_worker $N_WORKER --n_epoch $N_EPOCH \
--memory_size 1500 --transform "kd_trick" --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC

python main.py --mode efficient_memory --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET \
--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED \
--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--n_worker $N_WORKER --n_epoch $N_EPOCH \
--memory_size 1000 --transform "kd_trick" --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC

#python main.py --mode efficient_memory --mem_manage $MEM_MANAGE --exp_name $EXP \
#--dataset $DATASET \
#--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
#--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
#--rnd_seed $RND_SEED \
#--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
#--lr $LR --batchsize $BATCHSIZE \
#--n_worker $N_WORKER --n_epoch $N_EPOCH \
#--memory_size 500 --transform "kd_trick review specaug labels_trick" --uncert_metric $UNCERT_METRIC \
#--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC
#
#
#python main.py --mode efficient_memory --mem_manage $MEM_MANAGE --exp_name $EXP \
#--dataset $DATASET \
#--stream_env $STREAM $INIT_MODEL $INIT_OPT --topk $TOPK \
#--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
#--rnd_seed $RND_SEED \
#--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
#--lr $LR --batchsize $BATCHSIZE \
#--n_worker $N_WORKER --n_epoch $N_EPOCH \
#--memory_size 500 --transform "kd_trick mixup review specaug labels_trick" --uncert_metric $UNCERT_METRIC \
#--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC

#python main.py --mode efficient_memory --mem_manage $MEM_MANAGE --exp_name $EXP \
#--dataset $DATASET \
#--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
#--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
#--rnd_seed $RND_SEED \
#--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
#--lr $LR --batchsize $BATCHSIZE \
#--n_worker $N_WORKER --n_epoch $N_EPOCH \
#--memory_size 500 --transform "review" --uncert_metric $UNCERT_METRIC \
#--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC

