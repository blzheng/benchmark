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
        self.conv2d91 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x283):
        x284=self.conv2d91(x283)
        x285=self.sigmoid18(x284)
        return x285

m = M().eval()
x283 = torch.randn(torch.Size([1, 36, 1, 1]))
start = time.time()
output = m(x283)
end = time.time()
print(end-start)
