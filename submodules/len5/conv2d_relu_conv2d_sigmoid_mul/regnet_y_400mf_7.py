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
        self.conv2d41 = Conv2d(208, 52, kernel_size=(1, 1), stride=(1, 1))
        self.relu31 = ReLU()
        self.conv2d42 = Conv2d(52, 208, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()

    def forward(self, x128, x127):
        x129=self.conv2d41(x128)
        x130=self.relu31(x129)
        x131=self.conv2d42(x130)
        x132=self.sigmoid7(x131)
        x133=operator.mul(x132, x127)
        return x133

m = M().eval()
x128 = torch.randn(torch.Size([1, 208, 1, 1]))
x127 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x128, x127)
end = time.time()
print(end-start)
