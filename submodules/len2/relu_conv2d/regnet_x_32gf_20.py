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
        self.relu20 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x72):
        x73=self.relu20(x72)
        x74=self.conv2d23(x73)
        return x74

m = M().eval()
x72 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x72)
end = time.time()
print(end-start)
