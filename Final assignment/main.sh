wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.005 \
    --num-workers 10 \
    --seed 11 \
    --experiment-id "dinov2-unfrozen-lr0005"