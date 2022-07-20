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
        self.relu121 = ReLU(inplace=True)
        self.conv2d121 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x430):
        x431=self.relu121(x430)
        x432=self.conv2d121(x431)
        return x432

m = M().eval()
x430 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x430)
end = time.time()
print(end-start)
