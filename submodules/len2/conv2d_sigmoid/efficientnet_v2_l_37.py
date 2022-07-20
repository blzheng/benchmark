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
        self.conv2d221 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid37 = Sigmoid()

    def forward(self, x712):
        x713=self.conv2d221(x712)
        x714=self.sigmoid37(x713)
        return x714

m = M().eval()
x712 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x712)
end = time.time()
print(end-start)
