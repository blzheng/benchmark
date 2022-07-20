import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d55 = Conv2d(192, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x361, x354, x356, x357):
        x362=x361.view(x354, -1, x356, x357)
        x363=self.conv2d55(x362)
        x364=self.batchnorm2d55(x363)
        return x364

m = M().eval()
x361 = torch.randn(torch.Size([1, 96, 2, 7, 7]))
x354 = 1
x356 = 7
x357 = 7
start = time.time()
output = m(x361, x354, x356, x357)
end = time.time()
print(end-start)
