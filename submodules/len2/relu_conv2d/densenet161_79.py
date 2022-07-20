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
        self.relu80 = ReLU(inplace=True)
        self.conv2d80 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x285):
        x286=self.relu80(x285)
        x287=self.conv2d80(x286)
        return x287

m = M().eval()
x285 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x285)
end = time.time()
print(end-start)
