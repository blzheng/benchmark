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
        self.conv2d38 = Conv2d(480, 120, kernel_size=(1, 1), stride=(1, 1))
        self.relu14 = ReLU()
        self.conv2d39 = Conv2d(120, 480, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid3 = Hardsigmoid()

    def forward(self, x112):
        x113=self.conv2d38(x112)
        x114=self.relu14(x113)
        x115=self.conv2d39(x114)
        x116=self.hardsigmoid3(x115)
        return x116

m = M().eval()
x112 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x112)
end = time.time()
print(end-start)
