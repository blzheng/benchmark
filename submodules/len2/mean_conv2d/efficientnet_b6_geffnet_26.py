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
        self.conv2d130 = Conv2d(1200, 50, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x388):
        x389=x388.mean((2, 3),keepdim=True)
        x390=self.conv2d130(x389)
        return x390

m = M().eval()
x388 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x388)
end = time.time()
print(end-start)