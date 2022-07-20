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
        self.relu71 = ReLU()
        self.conv2d93 = Conv2d(308, 3024, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()

    def forward(self, x291):
        x292=self.relu71(x291)
        x293=self.conv2d93(x292)
        x294=self.sigmoid17(x293)
        return x294

m = M().eval()
x291 = torch.randn(torch.Size([1, 308, 1, 1]))
start = time.time()
output = m(x291)
end = time.time()
print(end-start)
