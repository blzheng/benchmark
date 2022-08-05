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
        self.conv2d20 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x66):
        x67=self.conv2d20(x66)
        return x67

m = M().eval()
x66 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x66)
end = time.time()
print(end-start)
