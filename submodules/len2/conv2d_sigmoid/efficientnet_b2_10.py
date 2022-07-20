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
        self.conv2d52 = Conv2d(22, 528, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()

    def forward(self, x158):
        x159=self.conv2d52(x158)
        x160=self.sigmoid10(x159)
        return x160

m = M().eval()
x158 = torch.randn(torch.Size([1, 22, 1, 1]))
start = time.time()
output = m(x158)
end = time.time()
print(end-start)
