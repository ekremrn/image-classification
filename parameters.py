import argparse



def get():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = 'data', type = str, help = 'Dataset to use')
    parser.add_argument('--dataset', default = 'CIFAR10', type = str, help = 'Dataset to use')

    parser.add_argument('--save_path', default = 'results', type = str, help = 'Model and log path')

    parser.add_argument('--epoch', default = 50, type = int, help = 'Epoch')
    parser.add_argument('--size', default = 32, type = int, help = 'Input image size')
    parser.add_argument('--batch_size', default = 264, type = int, help = 'Batch size')

    parser.add_argument('--aug_mode', default = "min", type = str, help = 'Augemntation modes: min, big')

    parser.add_argument('--arch', default = "resnet50", type = str, help = 'Could be used all the models in timm')
    parser.add_argument('--not_pretrained', action = 'store_false', help = 'Flag. If set, no ImageNet pretraining is used to initialize the network.')

    parser.add_argument('--optim', default = "sgd", type = str, help = 'Optimization method to use')
    parser.add_argument('--lr', default = 0.01, type = float, help = 'learning rate')
    parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum')

    parser.add_argument('--scheduler', default = "step", type = str, help = 'Optimization method to use')
    parser.add_argument('--lr_milestones', default = [10, 20, 30, 40], type = int, nargs = '+', help = 'lr milestones')

    parser.add_argument('--wandb_entity', default = "", type = str, help = 'keep it empty if wont use')

    parser.add_argument('--device', default = "cuda", type = str, help = 'device: cuda or cpu')
    parser.add_argument('--seed', default = 42, type = int, help = 'seed')
    parser.add_argument('--version', default = "", type = str, help = 'model version')

    return parser.parse_args()

