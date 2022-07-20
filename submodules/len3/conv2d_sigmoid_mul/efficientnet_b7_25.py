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
        self.conv2d125 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()

    def forward(self, x392, x389):
        x393=self.conv2d125(x392)
        x394=self.sigmoid25(x393)
        x395=operator.mul(x394, x389)
        return x395

m = M().eval()
x392 = torch.randn(torch.Size([1, 40, 1, 1]))
x389 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x392, x389)
end = time.time()
print(end-start)
