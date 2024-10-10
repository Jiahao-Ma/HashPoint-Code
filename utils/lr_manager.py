import abc
import numpy as np
# This code borrow from NeuRays; refer to https://github.com/liuyuan-pal/NeuRay.git

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class LearningRateManager(abc.ABC):
    @staticmethod
    def set_lr_for_all(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def construct_optimizer(self, optimizer, network):
        # may specify different lr for different parts
        # use group to set learning rate
        paras = network.parameters()
        return optimizer(paras, lr=1e-3)

    @abc.abstractmethod
    def __call__(self, optimizer, step, *args, **kwargs):
        pass

class ExpDecayLR(LearningRateManager):
    def __init__(self,cfg):
        self.lr_init=cfg['lr_init']
        self.decay_step=cfg['decay_step']
        self.decay_rate=cfg['decay_rate']
        self.lr_min=1e-5

    def __call__(self, optimizer, step, *args, **kwargs):
        lr=max(self.lr_init*(self.decay_rate**(step//self.decay_step)),self.lr_min)
        self.set_lr_for_all(optimizer,lr)
        return lr

class ExpDecayLRRayFeats(ExpDecayLR):
    def construct_optimizer(self, optimizer, network):
        paras = network.parameters()
        return optimizer([para for para in paras] + network.ray_feats, lr=1e-3)

class WarmUpExpDecayLR(LearningRateManager):
    def __init__(self, cfg):
        self.lr_warm=cfg['lr_warm']
        self.warm_step=cfg['warm_step']
        self.lr_init=cfg['lr_init']
        self.decay_step=cfg['decay_step']
        self.decay_rate=cfg['decay_rate']
        self.lr_min=1e-5

    def __call__(self, optimizer, step, *args, **kwargs):
        if step<self.warm_step:
            lr=self.lr_warm
        else:
            lr=max(self.lr_init*(self.decay_rate**((step-self.warm_step)//self.decay_step)),self.lr_min)
        self.set_lr_for_all(optimizer,lr)
        return lr

name2lr_manager={
    'exp_decay': ExpDecayLR,
    'exp_decay_ray_feats': ExpDecayLRRayFeats,
    'warm_up_exp_decay': WarmUpExpDecayLR,
}