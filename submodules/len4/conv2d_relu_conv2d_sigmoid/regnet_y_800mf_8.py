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
        self.conv2d46 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu35 = ReLU()
        self.conv2d47 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()

    def forward(self, x144):
        x145=self.conv2d46(x144)
        x146=self.relu35(x145)
        x147=self.conv2d47(x146)
        x148=self.sigmoid8(x147)
        return x148

m = M().eval()
x144 = torch.randn(torch.Size([1, 320, 1, 1]))
start = time.time()
output = m(x144)
end = time.time()
print(end-start)
