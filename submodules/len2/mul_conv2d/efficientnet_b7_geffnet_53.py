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
        self.conv2d266 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x789, x794):
        x795=operator.mul(x789, x794)
        x796=self.conv2d266(x795)
        return x796

m = M().eval()
x789 = torch.randn(torch.Size([1, 3840, 7, 7]))
x794 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x789, x794)
end = time.time()
print(end-start)
