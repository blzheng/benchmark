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
        self.relu88 = ReLU(inplace=True)
        self.conv2d93 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x305):
        x306=self.relu88(x305)
        x307=self.conv2d93(x306)
        return x307

m = M().eval()
x305 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x305)
end = time.time()
print(end-start)
