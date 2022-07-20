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
        self.conv2d151 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid30 = Sigmoid()

    def forward(self, x471):
        x472=self.conv2d151(x471)
        x473=self.sigmoid30(x472)
        return x473

m = M().eval()
x471 = torch.randn(torch.Size([1, 76, 1, 1]))
start = time.time()
output = m(x471)
end = time.time()
print(end-start)