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
        self.relu27 = ReLU()
        self.conv2d37 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()

    def forward(self, x113):
        x114=self.relu27(x113)
        x115=self.conv2d37(x114)
        x116=self.sigmoid6(x115)
        return x116

m = M().eval()
x113 = torch.randn(torch.Size([1, 80, 1, 1]))
start = time.time()
output = m(x113)
end = time.time()
print(end-start)
