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
        self.conv2d202 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x597, x602):
        x603=operator.mul(x597, x602)
        x604=self.conv2d202(x603)
        return x604

m = M().eval()
x597 = torch.randn(torch.Size([1, 2064, 7, 7]))
x602 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x597, x602)
end = time.time()
print(end-start)
