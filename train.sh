# CUDA_VISIBLE_DEVICES=7 python  main.py -c ./config/nir.yml --scale 8 --model_name Base2 --show_every 10 --epochs 40  --opt Adam --decay_epochs '40_70' --lr 3e-4 --embed_dim 64 --sched multistep --seed 60 --loss 1*L1

# CUDA_VISIBLE_DEVICES=7 python  main.py -c ./config/nir.yml --scale 8 --model_name Base2 --show_every 10 --epochs 90  --opt Adam --decay_epochs '40_70' --lr 3e-4 --embed_dim 64 --sched multistep --seed 60 --resume --load_name checkpoints/NIR/Base2_S_8_Loss_1*L1_LR_0.0003_Bs_8_Ps_256_Seed_60/model_000040.pth --loss 1*MSE


CUDA_VISIBLE_DEVICES=7 python  main.py -c ./config/nir.yml --scale 8 --model_name Base4 --show_every 10 --epochs 40  --opt Adam --decay_epochs '40_70' --lr 3e-4 --embed_dim 64 --sched multistep --seed 60 --loss 1*L1

CUDA_VISIBLE_DEVICES=7 python  main.py -c ./config/nir.yml --scale 8 --model_name Base4 --show_every 10 --epochs 90  --opt Adam --decay_epochs '40_70' --lr 3e-4 --embed_dim 64 --sched multistep --seed 60 --resume --load_name checkpoints/NIR/Base4_S_8_Loss_1*L1_LR_0.0003_Bs_8_Ps_256_Seed_60/model_000040.pth --loss 1*MSE