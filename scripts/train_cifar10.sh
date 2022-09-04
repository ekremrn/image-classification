python3 train.py --root data \
                 --dataset CIFAR10 \
                 --save_path results \
                 --epoch 25 \
                 --size 32 \
                 --batch_size 256 \
                 --aug_mode special \
                 --arch resnet50 \
                 --lr 0.005 \
                 --lr_milestones 5 10 15 20 \
                 --scheduler_gamma 0.5 \
                 --version 0.1

