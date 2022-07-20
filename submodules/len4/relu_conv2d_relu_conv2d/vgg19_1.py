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
        self.relu5 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x13):
        x14=self.relu5(x13)
        x15=self.conv2d6(x14)
        x16=self.relu6(x15)
        x17=self.conv2d7(x16)
        return x17

m = M().eval()
x13 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)
