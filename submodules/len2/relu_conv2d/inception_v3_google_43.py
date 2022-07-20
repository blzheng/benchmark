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
        self.conv2d81 = Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x276):
        x277=torch.nn.functional.relu(x276,inplace=True)
        x278=self.conv2d81(x277)
        return x278

m = M().eval()
x276 = torch.randn(torch.Size([1, 448, 5, 5]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
