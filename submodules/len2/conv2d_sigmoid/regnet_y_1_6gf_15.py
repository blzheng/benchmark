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
        self.conv2d82 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()

    def forward(self, x258):
        x259=self.conv2d82(x258)
        x260=self.sigmoid15(x259)
        return x260

m = M().eval()
x258 = torch.randn(torch.Size([1, 84, 1, 1]))
start = time.time()
output = m(x258)
end = time.time()
print(end-start)
