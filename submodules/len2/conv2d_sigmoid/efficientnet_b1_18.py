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
        self.conv2d92 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x282):
        x283=self.conv2d92(x282)
        x284=self.sigmoid18(x283)
        return x284

m = M().eval()
x282 = torch.randn(torch.Size([1, 48, 1, 1]))
start = time.time()
output = m(x282)
end = time.time()
print(end-start)
