import numpy as np

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