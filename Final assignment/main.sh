wandb login

export DINOV2_BASE_DIR="/gpfs/home2/scur2194/NNCV/Final assignment"

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 20 \
    --lr 0.00005 \
    --num-workers 10 \
    --seed 11 \
    --experiment-id "dinovits2-168x336-teacher"