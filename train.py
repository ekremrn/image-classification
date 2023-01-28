# LIBRARIES
##
import os
import json
import random
import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

##
import torch
import torch.nn as nn
from torch import optim
from sam import SAM
from sklearn.metrics import f1_score

##
import parameters
import dataset
import architecture

##LOG
import wandb
import logging
from utils import graph_loss_acc, simple_graph
from utils import plot_classification_report, plot_confusion_matrix

# INPUT ARGUMENTS
opt = parameters.get()


# LOG
model_name = "ds-{}_md-{}_in-{}".format(opt.dataset, opt.arch, opt.size)
if opt.version:
    model_name += "_v-{}".format(opt.version)

if opt.wandb_entity:
    wandb.init(
        entity=opt.wandb_entity, project=opt.dataset, config=opt, name=model_name
    )

opt.save_path = os.path.join(opt.save_path, model_name)
if not os.path.isdir(opt.save_path):
    os.makedirs(opt.save_path)

opt.train_graph_path = os.path.join(opt.save_path, "train_graph.png")
opt.valid_graph_path = os.path.join(opt.save_path, "valid_graph.png")
opt.lr_graph_path = os.path.join(opt.save_path, "lr_graph.png")
opt.valid_f1_macro_graph_path = os.path.join(opt.save_path, "valid_f1_macro.png")
opt.valid_f1_weighted_graph_path = os.path.join(opt.save_path, "valid_f1_weighted.png")
opt.valid_confusion_matrix_path = os.path.join(opt.save_path, "confussion_matrix.png")
opt.valid_classification_report_path = os.path.join(
    opt.save_path, "classification_report.png"
)

LOGFILE = os.path.join(opt.save_path, "console.log")
FORMATTER = "%(asctime)s | %(levelname)s | %(message)s"
FILEHANDLER = logging.FileHandler(LOGFILE)
STDOUTHANDLER = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO, format=FORMATTER, handlers=[FILEHANDLER, STDOUTHANDLER]
)
for arg, value in sorted(vars(opt).items()):
    logging.info("{}: {}".format(arg, value))

# SEED
torch.backends.cudnn.deterministic = True
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

# DATASET
transforms = dataset.get_transforms(opt.size, aug_mode=opt.aug_mode)
dataloaders = dataset.get(
    root=opt.root, dataset=opt.dataset, transforms=transforms, batch_size=opt.batch_size
)
opt.num_classes = dataloaders["num_classes"]
opt.classes = (
    dataloaders["training"].dataset.classes
    if hasattr(dataloaders["training"].dataset, "classes")
    else []
)

# MODEL CONFIG SAVE
model_config_dict = {
    "dataset": opt.dataset,
    "size": opt.size,
    "arch": opt.arch,
    "pretrained": opt.not_pretrained,
    "classes": opt.classes,
}
with open(os.path.join(opt.save_path, "config.json"), "w") as outfile:
    json.dump(model_config_dict, outfile)

# ARCHITECTURE
model = architecture.get(opt.arch, opt.not_pretrained, opt.num_classes, opt.device)


# LOSS FUNCTION AND OPTIMIZER

##LOSS
criterion = nn.CrossEntropyLoss()
criterion.to(opt.device)

##OPTIM
if opt.optim == "adam":
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
elif opt.optim == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
elif opt.optim == "sam":
    base_optimizer = optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, opt.lr, momentum=opt.momentum)
else:
    raise Exception("Optimizer <{}> not available!".format(opt.optim))

##SCHEDULER
if opt.scheduler == "step":
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer.base_optimizer if opt.optim == "sam" else optimizer,
        milestones=opt.step_lr_milestones,
        gamma=opt.step_scheduler_gamma,
    )
elif opt.scheduler == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer if opt.optim == "sam" else optimizer,
        T_max=opt.epoch,
        eta_min=opt.cosine_min_lr,
    )


# TRAIN
best_valid_acc = 0
best_valid_loss = 0
valid_loss_min = np.Inf

train_loss, train_acc, val_loss, val_acc = [], [], [], []
val_f1_macro, val_f1_weighted = [], []
learning_rate = []

total_train_step = len(dataloaders["training"])
total_test_step = len(dataloaders["validation"])

for epoch in range(opt.epoch):  # loop over the dataset multiple times

    ## Train Loop
    running_loss, correct, total = 0.0, 0, 0
    model.train()
    train_process = tqdm(
        enumerate(dataloaders["training"]),
        total=total_train_step,
        desc="Training Epoch: {}".format(epoch + 1),
        ncols=100,
    )

    for i, data in train_process:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(opt.device), labels.to(opt.device)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if opt.optim == "sam":
            optimizer.first_step(zero_grad=True)
            loss_ = criterion(model(inputs), labels)
            loss_.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        # zero the parameter gradients
        optimizer.zero_grad()

        # statistics
        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == labels).item()
        total += labels.size(0)

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_train_step)
    learning_rate.append(optimizer.param_groups[0]["lr"])

    ## Train Log
    logging.info(
        "Epoch: {}: train loss: {:.4f}, train acc: {:.4f}".format(
            epoch + 1, np.mean(train_loss), train_acc[-1]
        )
    )
    if opt.wandb_entity:
        wandb.log(
            {
                "train_loss": np.mean(train_loss),
                "train_acc": train_acc[-1],
                "learning rate": optimizer.param_groups[0]["lr"],
            }
        )
    graph_loss_acc(epoch + 1, train_loss, train_acc, opt.train_graph_path)
    simple_graph(
        epoch + 1,
        learning_rate,
        x_label="Epoch",
        y_label="LR",
        save_img=opt.lr_graph_path,
    )

    ## Test Loop
    running_loss, correct, total = 0.0, 0, 0
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        test_process = tqdm(
            enumerate(dataloaders["validation"]),
            total=total_test_step,
            desc="Validation ",
            ncols=100,
        )

        for i, data in test_process:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            y_true.extend(labels.data.cpu().numpy())

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            y_pred.extend(pred.data.cpu().numpy())
            correct += torch.sum(pred == labels).item()
            total += labels.size(0)

        val_acc.append(100 * correct / total)
        val_loss.append(running_loss / total_test_step)

        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        val_f1_macro.append(f1_macro)
        val_f1_weighted.append(f1_weighted)

        ## Test Log
        logging.info(
            "valid loss: {:.4f}, valid acc: {:.4f}\n".format(
                np.mean(val_loss), val_acc[-1]
            )
        )
        if opt.wandb_entity:
            wandb.log(
                {
                    "valid_loss": np.mean(val_loss),
                    "valid_acc": val_acc[-1],
                    "val_f1_macro": val_f1_macro[-1],
                    "val_f1_weighted": val_f1_weighted[-1],
                }
            )
        graph_loss_acc(epoch + 1, val_loss, val_acc, opt.valid_graph_path)
        simple_graph(
            epoch + 1,
            val_f1_macro,
            x_label="Epoch",
            y_label="F1 Macro",
            save_img=opt.valid_f1_macro_graph_path,
        )
        simple_graph(
            epoch + 1,
            val_f1_weighted,
            x_label="Epoch",
            y_label="F1 Weighted",
            save_img=opt.valid_f1_weighted_graph_path,
        )

        ## Save Models
        network_learned = running_loss < valid_loss_min
        if network_learned:
            valid_loss_min = running_loss
            plot_classification_report(
                y_true, y_pred, opt.classes, opt.valid_classification_report_path
            )
            plot_confusion_matrix(
                y_true, y_pred, opt.classes, opt.valid_confusion_matrix_path
            )
            torch.save(model.state_dict(), os.path.join(opt.save_path, "best.pt"))
            logging.info("Detected network improvement new model saved\n")

        torch.save(model.state_dict(), os.path.join(opt.save_path, "last.pt"))

    if opt.scheduler != "none":
        scheduler.step()
