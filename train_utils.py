import torch
import numpy as np
import math

def update_learning_rate(optimizer, iteration, position_lr_max_steps = 1000 * 10, spatial_lr_scale = 1.0) -> float:
    """
    Updates the learning rate for the optimizer based on the current iteration and maximum steps.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs to be updated.
        iteration (int): The current iteration or step in the training process.
        position_lr_max_steps (int): The maximum number of steps for the learning rate schedule.
    Returns:
        float: The updated learning rate for the 'xyz' parameter group.
    This function uses an exponential learning rate schedule with optional delay. The learning rate
    is calculated using the `expon_lr` function, which interpolates between an initial and final
    learning rate over the specified number of steps. If a delay is specified, the learning rate
    will be adjusted accordingly during the delay period.
    """
    def expon_lr(step, lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    for param_group in optimizer.param_groups:
        if param_group["name"] == "xyz":
            lr = expon_lr(
                step=iteration,
                lr_init=0.00016 * spatial_lr_scale,
                lr_final=0.0000016 * spatial_lr_scale,
                lr_delay_steps=0,
                lr_delay_mult=0.01,
                max_steps=position_lr_max_steps,
            )
            param_group["lr"] = lr
            return lr

def ssim(img1, img2, window_size=11, size_average=True):
    def create_window(window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [
                    math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                    for x in range(window_size)
                ]
            )
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(
            _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        )
        return window

    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = torch.nn.functional.conv2d(
            img1, window, padding=window_size // 2, groups=channel
        )
        mu2 = torch.nn.functional.conv2d(
            img2, window, padding=window_size // 2, groups=channel
        )
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = (
            torch.nn.functional.conv2d(
                img1 * img1, window, padding=window_size // 2, groups=channel
            )
            - mu1_sq
        )
        sigma2_sq = (
            torch.nn.functional.conv2d(
                img2 * img2, window, padding=window_size // 2, groups=channel
            )
            - mu2_sq
        )
        sigma12 = (
            torch.nn.functional.conv2d(
                img1 * img2, window, padding=window_size // 2, groups=channel
            )
            - mu1_mu2
        )
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
