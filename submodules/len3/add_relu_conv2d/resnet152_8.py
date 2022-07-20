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
        self.relu25 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x96, x88):
        x97=operator.add(x96, x88)
        x98=self.relu25(x97)
        x99=self.conv2d30(x98)
        return x99

m = M().eval()
x96 = torch.randn(torch.Size([1, 512, 28, 28]))
x88 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x96, x88)
end = time.time()
print(end-start)
