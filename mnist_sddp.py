#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm

import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

formatter = logging.Formatter("%(asctime)s - %(name)s %(levelname)s - %(message)s")

def init_logger(logger_name="mylogger"):
    """
    直接引入logging，创建一个对象
    """
    # 创建一个logger，名称为logger_name
    logger = logging.getLogger(logger_name)
    # 设定logger的默认级别
    logger.setLevel(logging.DEBUG)

    # 需要先判断句柄是否已经被创建，如果当前句柄为空，则可以添加
    if not logger.handlers:
        logger.addHandler(init_fileHd(formatter))
        logger.addHandler(init_streamHd(formatter))

    return logger


def init_fileHd(myformatter, log_file_name="mnist_kd082901.log"):
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(myformatter)
    return file_handler


def init_streamHd(myformatter):
    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(myformatter)
    return stream_handler

mylogger = init_logger("mylogger")

def parser_fun():
    parser = argparse.ArgumentParser(
            description="Opacus MNIST Example",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=400,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
            "-r",
            "--n-runs",
            type=int,
            default=1,
            metavar="R",
            help="number of runs to average on",
        )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=4.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
            "--save-model",
            action="store_true",
            default=False,
            help="Save the trained model",
        )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    return args,device
args,device = parser_fun()
temperature = 2
mylogger.info(f"教师模型和学生模型同时训练,模型用了relu激活函数")
mylogger.info(f"kd,temperature={temperature},ce_weight=0.1,kd_weight=0.9")

mylogger.info(f"batch_size:{args.batch_size}," \
              f" lr:{args.lr},epochs:{args.epochs}" \
              f" delta:{args.delta}, sigma:{args.sigma}," \
              f" max-per-sample-grad_norm:{args.max_per_sample_grad_norm}"
              )

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        # tanh;selu;relu
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

def test_same(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTeacher Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch, teacher_model, teacher_optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()

    teacher_model.train()
    teacher_criterion = nn.CrossEntropyLoss()

    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        teacher_optimizer.zero_grad()
        optimizer.zero_grad()

        teacher_output = teacher_model(data)
        teacher_output_kd = teacher_output.detach()
        output = model(data)

        teacher_loss = teacher_criterion(teacher_output, target)
        loss_ce = criterion(output, target)
        loss_kd = 0.9 * kd_loss(
            output, teacher_output_kd, temperature
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        loss = sum([l.mean() for l in losses_dict.values()])

        teacher_loss.backward()
        loss.backward()

        teacher_optimizer.step()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
        return epsilon
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
        return 0

def main():

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = []
    for _ in range(args.n_runs):

        # 教师模型和教师模型对应的优化器
        teacher_model = SampleConvNet().to(device)

        teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=args.lr, momentum=0)

        # 学生模型和学生模型对应的优化器
        model = SampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

        epoch_train_acc = []
        epoch_test_acc = []
        epoch_epsilon = []
        for epoch in range(1, args.epochs + 1):
            epoch_epsilon.append(train(args, model, device, train_loader, optimizer, privacy_engine, epoch,
                                       teacher_model, teacher_optimizer))
            epoch_train_acc.append(test_same(model, device, train_loader))
            epoch_test_acc.append(test_same(model, device, test_loader))
        mylogger.info(f"epoch_epsilon:{epoch_epsilon}")
        mylogger.info(f"epoch_train_acc:{epoch_train_acc}")
        mylogger.info(f"epoch_test_acc:{epoch_test_acc}")
        #mylogger.info(f"epoch_acc:{epoch_acc}")
        #run_results.append(mytest(model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    # repro_str = (
    #     f"mnist_{args.lr}_{args.sigma}_"
    #     f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    # )
    # torch.save(run_results, f"run_results_{repro_str}.pt")
    #
    # if args.save_model:
    #     torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()
