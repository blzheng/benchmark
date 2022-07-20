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
        self.relu122 = ReLU(inplace=True)
        self.conv2d122 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x432):
        x433=self.relu122(x432)
        x434=self.conv2d122(x433)
        return x434

m = M().eval()
x432 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x432)
end = time.time()
print(end-start)
