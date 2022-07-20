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
        self.relu12 = ReLU()
        self.conv2d44 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid7 = Hardsigmoid()

    def forward(self, x126, x124):
        x127=self.relu12(x126)
        x128=self.conv2d44(x127)
        x129=self.hardsigmoid7(x128)
        x130=operator.mul(x129, x124)
        return x130

m = M().eval()
x126 = torch.randn(torch.Size([1, 144, 1, 1]))
x124 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x126, x124)
end = time.time()
print(end-start)
