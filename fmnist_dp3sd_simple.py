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
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
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


def load_data_fashionmnist(args, train_batch_size):

    FASHION_MNIST_MEAN = 0.286
    FASHION_MNIST_STD = 0.3529

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
                ]
            ),
        ),
        # batch_size=args.batch_size,
        batch_size=train_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader


def set_random_seed(seed_value):
    import random
    torch.manual_seed(seed_value)  # 设置 PyTorch 随机种子
    torch.cuda.manual_seed_all(seed_value)  # 如果使用多个 GPU，也应该添加这行
    np.random.seed(seed_value)  # 设置 NumPy 的随机种子
    random.seed(seed_value)  # 设置 Python 原生随机库的种子
    torch.backends.cudnn.deterministic = True  # 确保 CNN 的一致性
    torch.backends.cudnn.benchmark = False  # 可提高训练速度，但在不同运行中可能会导致微小差异


def main():
    args = fashionmnist_parse_args()
    device = torch.device(args.device)

    set_random_seed(42)

    lr_z = 3
    batch_size_z = 1600
    norm_z = .1  # 裁剪阈值
    # sigma_zdf = [1.2]

    epochs_z = 60
    epsilon_z = 3.0

    temperature_t = 5  # smooth temperature
    temperature_s = 0.3  # sparse temperature

    # alpha 可用于平衡 loss
    dkd_alpha = 0.3

    # 此处用到了batch_size_z
    train_loader, test_loader = load_data_fashionmnist(args, batch_size_z)

    best_acc1 = 0
    best_epsilon = 0
    best_epoch = 0

    model = SampleConvNet().to(device)
    teacher_model = SampleConvNet().to(device)


    # 此处用到了 lr_z
    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            # lr=args.lr,
            lr=lr_z,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        teacher_optimizer = optim.SGD(
            teacher_model.parameters(),
            # lr=args.lr,
            lr=lr_z,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    privacy_engine = None
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            secure_mode=args.secure_rng,
        )
        clipping = "per_layer" if args.clip_per_layer else "flat"

        # 此处用到了norm_z and epsilon_z, epochs_z
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs_z,
            target_epsilon=epsilon_z,
            max_grad_norm=norm_z,
            clipping=clipping,
            grad_sample_mode=args.grad_sample_mode,
            target_delta=1e-5
        )

        teacher_model, teacher_optimizer, teacher_train_loader = privacy_engine.make_private_with_epsilon(
            module=teacher_model,
            optimizer=teacher_optimizer,
            data_loader=train_loader,
            epochs=epochs_z,
            target_epsilon=epsilon_z,
            max_grad_norm=norm_z,
            clipping=clipping,
            grad_sample_mode=args.grad_sample_mode,
            target_delta=1e-5
        )

    # Store some logs
    accuracy_per_epoch_test = []
    accuracy_per_epoch_train = []
    epsilon_accumulate = []

    for epoch in range(0, epochs_z):

        if epoch > 0:
            teacher_model.load_state_dict(model.state_dict())
            temperature_s_epoch = temperature_s + epoch / epochs_z * (
                        1 - temperature_s)

            train_duration, epsilon_epoch = train_sd_sparse2(
                args, model, optimizer, teacher_model,
                train_loader,
                privacy_engine, epoch, device,
                dkd_alpha,
                temperature_t,
                #temperature_s
                temperature_s_epoch
            )
        else:
            # train dpsgd
            train_duration, epsilon_epoch = train_dpsgd(
                args, model,
                optimizer,
                train_loader,
                privacy_engine,
                epoch, device)

        print(f"epoch:{epoch}")
        top1_acc, test_loss = test(args, model, test_loader, device, True)
        top1_acc_train, train_loss = test(args, model, train_loader, device, False)

        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)
        if is_best:
            best_epsilon = epsilon_epoch
            best_epoch = epoch
        epsilon_accumulate.append(epsilon_epoch)

        accuracy_per_epoch_test.append(float(top1_acc))
        accuracy_per_epoch_train.append(float(top1_acc_train))

        # mnist_logits_plt(test_loader, model, epoch, "dpdkd")

    print(f"best_accuracy: {best_acc1}")
    print(f"best_epsilon: {best_epsilon}")
    print(f"best_epoch: {best_epoch}")


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def train_dpsgd(args, model, optimizer,
                train_loader,
                privacy_engine, epoch, device):
    start_time = datetime.now()

    model.train()
    criterion = nn.CrossEntropyLoss()

    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

    train_duration = datetime.now() - start_time

    if not args.disable_dp:
        epsilon_epoch = privacy_engine.accountant.get_epsilon(delta=args.delta)
    else:
        epsilon_epoch = 0

    return train_duration, epsilon_epoch


def train_sd_sparse(args, model, optimizer, teacher_model,
                    train_loader,
                    privacy_engine, epoch, device,
                    sd_coff,
                    temperature_teacher,
                    temperature_student):
    start_time = datetime.now()

    model.train()
    criterion = nn.CrossEntropyLoss()  # hard label loss

    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        # CE使用稀疏性
        # CE使用稀疏性
        if epoch < 5:
            output = output / temperature_student
        loss_ce = criterion(output, target)
        # if epoch < 10:
        #     output_sparse = output / temperature_student
        # else:
        #     output_sparse = output
        #
        # loss_ce = criterion(output_sparse, target)

        with torch.no_grad():
            teacher_output = teacher_model(images)
            teacher_output_kd = teacher_output.detach()

        sd_loss_value = kd_loss(output, teacher_output_kd, temperature_teacher)

        loss = loss_ce + sd_coff * sd_loss_value

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

    train_duration = datetime.now() - start_time
    # epsilon_epoch = privacy_engine.accountant.get_epsilon(delta=args.delta)
    if not args.disable_dp:
        epsilon_epoch = privacy_engine.accountant.get_epsilon(delta=args.delta)
    else:
        epsilon_epoch = 0
    # epsilon_epoch = privacy_engine.get_epsilon(delta=args.delta)

    return train_duration, epsilon_epoch


def test(args, model, test_loader, device, flag=True):
    """
    flag=True 表明是测试数据集，Fasle 代表的是训练数据集
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0

    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            # 第一种计算方法
            # preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            # labels = target.detach().cpu().numpy()
            # acc1 = accuracy(preds, labels)
            # 第二种计算方法
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            losses.append(loss.item())

    top1_avg2 = correct / len(test_loader.dataset)

    # print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    if flag:
        print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " + f"Acc@1: {top1_avg2 :.6f}")
    else:
        print(f"\tTrain set:" f"Loss: {np.mean(losses):.6f} " + f"Acc@1: {top1_avg2 :.6f}")

    # return np.mean(top1_acc)
    return top1_avg2, losses


def fashionmnist_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus fashionMNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="../fashionmnist",
        help="Where fashionMNIST is/will be stored",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1600,
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
        default=60,
        metavar="N",
        help="number of epochs to train",
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
        "--lr",
        type=float,
        default=0.3,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        metavar="AL",
        help="coff of NCSD",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        metavar="BE",
        help="coff of TCSD",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2,
        metavar="T",
        help="temperature for soft label",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        metavar="CO",
        help="gamma for teacher",
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
        default=1.0,
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
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )

    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )
    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )
    parser.add_argument("--grad_sample_mode", type=str, default="hooks")
    return parser.parse_args()


def train_sd_sparse2(args, model, optimizer, teacher_model,
                    train_loader,
                    privacy_engine, epoch, device,
                    sd_coff,
                    temperature_teacher,
                    temperature_student):
    start_time = datetime.now()

    model.train()
    criterion = nn.CrossEntropyLoss()  # hard label loss

    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        # CE使用稀疏性
        if epoch < 5:
            output = output / temperature_student

        loss_ce = criterion(output, target)

        with torch.no_grad():
            teacher_output = teacher_model(images)
            teacher_output_kd = teacher_output.detach()

        sd_loss_value = kd_loss(output, teacher_output_kd, temperature_teacher)
        # sd_loss_value = kd_loss_sparse(output, teacher_output_kd,
        #                                temperature_student, temperature_teacher)

        # loss = (1 - sd_coff) * loss_ce + sd_coff * sd_loss_value
        loss = loss_ce + sd_coff * sd_loss_value

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

    train_duration = datetime.now() - start_time
    # epsilon_epoch = privacy_engine.accountant.get_epsilon(delta=args.delta)
    if not args.disable_dp:
        epsilon_epoch = privacy_engine.accountant.get_epsilon(delta=args.delta)
    else:
        epsilon_epoch = 0
    # epsilon_epoch = privacy_engine.get_epsilon(delta=args.delta)

    return train_duration, epsilon_epoch

if __name__ == "__main__":
    main()
