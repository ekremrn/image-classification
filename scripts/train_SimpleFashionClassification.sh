python3 train.py --dataset SimpleFashionClassification \
                 --epoch 25 \
                 --size 128 \
                 --batch_size 64 \
                 --aug_mode min \
                 --arch resnet50


python3 train.py --root data \
                 --dataset SimpleFashionClassification \
                 --epoch 25 \
                 --size 128 \
                 --batch_size 64 \
                 --aug_mode big \
                 --arch resnet50


python3 train.py --root data \
                 --dataset SimpleFashionClassification \
                 --epoch 25 \
                 --size 264 \
                 --batch_size 64 \
                 --aug_mode min \
                 --arch resnet50


python3 train.py --root data \
                 --dataset SimpleFashionClassification \
                 --epoch 25 \
                 --size 264 \
                 --batch_size 64 \
                 --aug_mode big \
                 --arch resnet50
