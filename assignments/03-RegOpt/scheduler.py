from typing import List

from torch.optim.lr_scheduler import _LRScheduler

import math


class CustomLRScheduler(_LRScheduler):
    """
    customer class for LRScheduler
    """

    def __init__(
        self,
        optimizer,
        # step_size,
        # T_max,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        # gamma=0.9,
        # total_iters=5,
        # power=1.0,
        verbose=False,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        # self.step_size = step_size
        # self.gamma = gamma
        # self.total_iters = total_iters
        # self.power = power
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.current_T = T_0
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.end_of_cycle = False
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """
        1. use gamma to adjust lr
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]

        # if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
        #     return [group["lr"] for group in self.optimizer.param_groups]
        # return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

        # if self.last_epoch == 0 or self.last_epoch > self.total_iters:
        #     return [group["lr"] for group in self.optimizer.param_groups]
        #
        # decay_factor = (
        #     (1.0 - self.last_epoch / self.total_iters)
        #     / (1.0 - (self.last_epoch - 1) / self.total_iters)
        # ) ** self.power
        # return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

        # if self.last_epoch == 0:
        #     return [group["lr"] for group in self.optimizer.param_groups]
        # elif self._step_count == 1 and self.last_epoch > 0:
        #     return [
        #         self.eta_min
        #         + (base_lr - self.eta_min)
        #         * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
        #         / 2
        #         for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        #     ]
        # elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
        #     return [
        #         group["lr"]
        #         + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
        #         for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        #     ]
        # return [
        #     (1 + math.cos(math.pi * self.last_epoch / self.T_max))
        #     / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
        #     * (group["lr"] - self.eta_min)
        #     + self.eta_min
        #     for group in self.optimizer.param_groups
        # ]

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.end_of_cycle = self.last_epoch % self.current_T == self.current_T - 1
        super(CustomLRScheduler, self).step(epoch)
