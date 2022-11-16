import os
import torch
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime
import time
from omegaconf import OmegaConf
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim import lr_scheduler

######
from model import (
    TIMM,
    LabPreNorm,
    LabEMAPreNorm,
    LabRandNorm,
)
from set import HistoDataset
from utils import (
    AverageMeter,
    accuracy,
    save_log,
    LOGITS,
)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
# To fix the EOFError,discribed in https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0

train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class Trainer:
    def __init__(
        self,
        config_path: str,
    ):
        config = OmegaConf.load(config_path)

        if hasattr(config, "seed"):
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        ##### Create Dataloaders.
        trainset = HistoDataset(
            root=config.train_root,
            transform=train_transform,
        )
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        testset = HistoDataset(
            root=config.test_root,
            transform=test_transform,
        )
        test_loader = DataLoader(
            dataset=testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

        num_classes = len(os.listdir(config.train_root))

        ##### Create folders for the outputs.
        postfix = time.strftime("%Y%m%d_%H:%M") + "_" + config.model
        if hasattr(config, "postfix") and config.postfix != "":
            postfix += "_" + config.postfix

        self.output_path = os.path.join(config.output_path, postfix)

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "weights"), exist_ok=True)
        self.logging = open(os.path.join(self.output_path, "logging.txt"), "w+")

        OmegaConf.save(config=config, f=os.path.join(self.output_path, "config.yaml"))
        ##### Create models.
        if hasattr(config, "gpu_id"):
            self.device = torch.device(
                "cuda:{}".format(config.gpu_id) if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TIMM(
            model_name=config.model,
            num_classes=num_classes,
        )

        if hasattr(config, "prenorm") and config.prenorm:
            print("Using PreNorm.")
            prenorm = True
            model = LabPreNorm(model, self.device)
        else:
            prenorm = False

        if hasattr(config, "emaprenorm") and config.emaprenorm:
            print("Using EMAPreNorm.")
            model = LabEMAPreNorm(
                model=model,
                device=self.device,
                lmbd=config.emaprenorm_lambda
                if hasattr(config, "emaprenorm_lambda")
                else 0,
            )

        if hasattr(config, "randnorm") and config.randnorm:
            print("Using RandNorm.")
            model = LabRandNorm(model, self.device)

        self.model = model.to(self.device)

        if not prenorm:
            self.optimizer = AdamW(
                params=self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = AdamW(
                params=[
                    {"params": model.mu, "lr": config.learning_rate / 10},
                    {
                        "params": model.sigma,
                        "lr": config.learning_rate / 50,
                    },
                    {"params": self.model.model.parameters()},
                ],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        if config.scheduler.lower() == "epoential":
            self.scheduler = lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=config.gamma
            )
        elif config.scheduler.lower() == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=config.T_max,
                eta_min=config.min_learning_rate,
            )
        elif config.scheduler.lower() == "constant":
            self.scheduler = lr_scheduler.ConstantLR(
                optimizer=self.optimizer,
            )
        else:
            raise ValueError("Unkown scheduler {}".format(config.scheduler.lower()))

        self.epochs = config.epochs
        self.patience = config.patience

    def train(
        self,
    ):
        best_epoch = 0.0
        best_test_acc = 0.0

        time_start = time.time()

        msg = "[{}] Total training epochs : {}".format(
            datetime.now().strftime("%A %H:%M"), self.epochs
        )
        save_log(self.logging, msg)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            test_loss, test_acc = self.test_per_epoch(model=self.model)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.output_path, "weights", "model_epoch{}.pth".format(epoch)
                    ),
                )
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_path, "weights", "best_model.pth"),
                )

            msg = "[{}] Epoch {:03d} \
                \n Train loss: {:.5f},   Train acc: {:.3f}%;\
                \n Test loss: {:.5f},   Test acc: {:.3f}%;  \
                \n Best test acc: {:.3f} \n".format(
                datetime.now().strftime("%A %H:%M"),
                epoch,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                best_test_acc,
            )
            save_log(self.logging, msg)

            if (epoch - best_epoch) > self.patience:
                break

        msg = "[{}] Best test acc:{:.3f}% @ epoch {} \n".format(
            datetime.now().strftime("%A %H:%M"), best_test_acc, best_epoch
        )
        save_log(self.logging, msg)

        time_end = time.time()
        msg = "[{}] run time: {:.1f}s, {:.2f}h\n".format(
            datetime.now().strftime("%A %H:%M"),
            time_end - time_start,
            (time_end - time_start) / 3600,
        )
        save_log(self.logging, msg)

    def train_one_epoch(self):
        train_loss_recorder = AverageMeter()
        train_acc_recorder = AverageMeter()

        self.model.train()

        for img, label in tqdm(self.train_loader):
            self.optimizer.zero_grad()

            img = img.to(self.device)
            label = label.to(self.device)

            out = self.model(img)[LOGITS]
            loss = F.cross_entropy(out, label)

            loss.backward()
            self.optimizer.step()

            acc = accuracy(out, label)[0]
            train_loss_recorder.update(loss.item(), out.size(0))
            train_acc_recorder.update(acc.item(), out.size(0))

        self.scheduler.step()

        train_loss = train_loss_recorder.avg
        train_acc = train_acc_recorder.avg

        return train_loss, train_acc

    def test_per_epoch(self, model):
        test_loss_recorder = AverageMeter()
        test_acc_recorder = AverageMeter()

        with torch.no_grad():
            model.eval()

            for img, label in tqdm(self.test_loader):

                img = img.to(self.device)
                label = label.to(self.device)

                out = self.model(img)[LOGITS]
                loss = F.cross_entropy(out, label)

                acc = accuracy(out, label)[0]

                test_loss_recorder.update(loss.item(), out.size(0))
                test_acc_recorder.update(acc.item(), out.size(0))

        test_loss = test_loss_recorder.avg
        test_acc = test_acc_recorder.avg

        return test_loss, test_acc
