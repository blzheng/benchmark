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
        self.conv2d55 = Conv2d(464, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x361, x354, x356, x357):
        x362=x361.view(x354, -1, x356, x357)
        x363=self.conv2d55(x362)
        return x363

m = M().eval()
x361 = torch.randn(torch.Size([1, 232, 2, 7, 7]))
x354 = 1
x356 = 7
x357 = 7
start = time.time()
output = m(x361, x354, x356, x357)
end = time.time()
print(end-start)
