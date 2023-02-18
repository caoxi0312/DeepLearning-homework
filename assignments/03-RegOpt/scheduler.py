from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    customer class for LRScheduler
    """

    def __init__(
        self,
        optimizer,
        # step_size,
        last_epoch=-1,
        # gamma=0.9,
        total_iters=5,
        power=1.0,
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
        self.total_iters = total_iters
        self.power = power
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

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = (
            (1.0 - self.last_epoch / self.total_iters)
            / (1.0 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]
