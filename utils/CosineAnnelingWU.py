import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import os
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.optim as optim


class CosineAnnealingWarmupRestarts2(optim.lr_scheduler._LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size 第一个循环周期的epoch 数量
        self.cycle_mult = cycle_mult  # cycle steps magnification # 之后循环周期增加比例
        self.base_max_lr = max_lr  # first max learning rate  # 最大的学习率
        self.max_lr = max_lr  # max learning rate in the current cycle # 当前最大学习率
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size   #
        self.gamma = gamma  # decrease rate of max learning rate by cycle   # 最大学习率衰减比例

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts2, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



if __name__ == "__main__":
    model = resnet18(pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    mode = 'cosineAnnWarm'
    if mode == 'cosineAnn':
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    elif mode == 'cosineAnnWarm':
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
        scheduler = CosineAnnealingWarmupRestarts2(optimizer,
                                                   first_cycle_steps=75,
                                                   cycle_mult=1,
                                                   max_lr=0.1,
                                                   min_lr=0.001,
                                                   warmup_steps=5,
                                                   gamma=0.1,
                                                   )
    plt.figure()
    max_epoch = 150
    iters = 64
    cur_lr_list = []
    for epoch in range(max_epoch):
        print('epoch_{}'.format(epoch))
        for batch in range(iters):
            scheduler.step(epoch + batch / iters)  # 2 + 3/5  更新学习率    iters : 总迭代数量， batch 当前迭代次数。
            optimizer.step()
            # scheduler.step()
            cur_lr = optimizer.param_groups[-1]['lr']
            cur_lr_list.append(cur_lr)
            print('cur_lr:', cur_lr)
        print('epoch_{}_end'.format(epoch))
    x_list = list(range(len(cur_lr_list)))
    plt.plot(x_list, cur_lr_list)
    plt.show()
