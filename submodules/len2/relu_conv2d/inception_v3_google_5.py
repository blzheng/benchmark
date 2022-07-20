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
        self.conv2d10 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x44):
        x45=torch.nn.functional.relu(x44,inplace=True)
        x46=self.conv2d10(x45)
        return x46

m = M().eval()
x44 = torch.randn(torch.Size([1, 96, 25, 25]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
