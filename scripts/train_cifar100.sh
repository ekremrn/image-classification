python3 train.py --root data \
                 --dataset CIFAR100 \
                 --save_path results \
                 --epoch 25 \
                 --size 32 \
                 --batch_size 64 \
                 --aug_mode min \
                 --arch resnet50 \
                 --wandb_entity ekrem-rn \
                 --version 0.1
                 

python3 train.py --root data \
                 --dataset CIFAR100 \
                 --save_path results \
                 --epoch 25 \
                 --size 32 \
                 --batch_size 64 \
                 --aug_mode big \
                 --arch resnet50 \
                 --wandb_entity ekrem-rn \
                 --version 0.2

python3 train.py --root data \
                 --dataset CIFAR100 \
                 --save_path results \
                 --epoch 25 \
                 --size 32 \
                 --batch_size 64 \
                 --aug_mode big \
                 --arch resnet50 \
                 --lr 0.001 \
                 --wandb_entity ekrem-rn \
                 --version 0.3


