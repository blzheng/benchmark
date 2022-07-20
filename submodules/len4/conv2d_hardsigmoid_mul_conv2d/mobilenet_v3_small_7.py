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
        self.conv2d44 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid7 = Hardsigmoid()
        self.conv2d45 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x127, x124):
        x128=self.conv2d44(x127)
        x129=self.hardsigmoid7(x128)
        x130=operator.mul(x129, x124)
        x131=self.conv2d45(x130)
        return x131

m = M().eval()
x127 = torch.randn(torch.Size([1, 144, 1, 1]))
x124 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x127, x124)
end = time.time()
print(end-start)
