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
        self.conv2d134 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x422):
        x423=self.conv2d134(x422)
        return x423

m = M().eval()
x422 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x422)
end = time.time()
print(end-start)
