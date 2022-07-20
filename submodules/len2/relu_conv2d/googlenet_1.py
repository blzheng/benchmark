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
        self.conv2d5 = Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x29):
        x30=torch.nn.functional.relu(x29,inplace=True)
        x31=self.conv2d5(x30)
        return x31

m = M().eval()
x29 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
