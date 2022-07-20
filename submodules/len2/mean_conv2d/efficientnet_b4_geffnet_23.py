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
        self.conv2d116 = Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x344):
        x345=x344.mean((2, 3),keepdim=True)
        x346=self.conv2d116(x345)
        return x346

m = M().eval()
x344 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x344)
end = time.time()
print(end-start)
