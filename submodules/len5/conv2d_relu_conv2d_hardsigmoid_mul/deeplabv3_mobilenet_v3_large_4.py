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
        self.conv2d43 = Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
        self.relu15 = ReLU()
        self.conv2d44 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid4 = Hardsigmoid()

    def forward(self, x128, x127):
        x129=self.conv2d43(x128)
        x130=self.relu15(x129)
        x131=self.conv2d44(x130)
        x132=self.hardsigmoid4(x131)
        x133=operator.mul(x132, x127)
        return x133

m = M().eval()
x128 = torch.randn(torch.Size([1, 672, 1, 1]))
x127 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x128, x127)
end = time.time()
print(end-start)
