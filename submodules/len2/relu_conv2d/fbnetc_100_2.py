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
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)

    def forward(self, x15):
        x16=self.relu3(x15)
        x17=self.conv2d5(x16)
        return x17

m = M().eval()
x15 = torch.randn(torch.Size([1, 96, 112, 112]))
start = time.time()
output = m(x15)
end = time.time()
print(end-start)
