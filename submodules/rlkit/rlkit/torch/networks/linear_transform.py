from rlkit.torch.core import PyTorchModule


class LinearTransform(PyTorchModule):
    def __init__(self, m, b):
        super().__init__()
        self.m = m
        self.b = b

    def __call__(self, t):
        print("Is this __call__for get_actions? this is linear_transform.py")
        return self.m * t + self.b
