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
        self.relu26 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x92):
        x93=self.relu26(x92)
        x94=self.conv2d29(x93)
        return x94

m = M().eval()
x92 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x92)
end = time.time()
print(end-start)
