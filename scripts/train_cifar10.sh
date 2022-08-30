python3 train.py --root data \
                 --dataset CIFAR10 \
                 --save_path results \
                 --epoch 25 \
                 --size 32 \
                 --batch_size 64 \
                 --aug_mode min \
                 --arch resnet18


python3 train.py --root data \
                 --dataset CIFAR10 \
                 --save_path results \
                 --epoch 25 \
                 --size 32 \
                 --batch_size 64 \
                 --aug_mode min \
                 --arch resnet50
                 

python3 train.py --root data \
                 --dataset CIFAR10 \
                 --save_path results \
                 --epoch 25 \
                 --size 32 \
                 --batch_size 64 \
                 --aug_mode big \
                 --arch resnet18
                 

python3 train.py --root data \
                 --dataset CIFAR10 \
                 --save_path results \
                 --epoch 25 \
                 --size 32 \
                 --batch_size 64 \
                 --aug_mode big \
                 --arch resnet50
                 