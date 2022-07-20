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
        self.conv2d193 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x603, x588):
        x604=operator.add(x603, x588)
        x605=self.conv2d193(x604)
        return x605

m = M().eval()
x603 = torch.randn(torch.Size([1, 512, 7, 7]))
x588 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x603, x588)
end = time.time()
print(end-start)
