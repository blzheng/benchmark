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
        self.relu150 = ReLU(inplace=True)
        self.conv2d150 = Conv2d(1120, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x532):
        x533=self.relu150(x532)
        x534=self.conv2d150(x533)
        return x534

m = M().eval()
x532 = torch.randn(torch.Size([1, 1120, 7, 7]))
start = time.time()
output = m(x532)
end = time.time()
print(end-start)
