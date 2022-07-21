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
        self.conv2d4 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x10):
        x11=self.conv2d4(x10)
        x12=self.relu4(x11)
        x13=self.conv2d5(x12)
        x14=self.relu5(x13)
        x15=self.conv2d6(x14)
        return x15

m = M().eval()
x10 = torch.randn(torch.Size([1, 128, 56, 56]))
start = time.time()
output = m(x10)
end = time.time()
print(end-start)
