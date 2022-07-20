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
        self.conv2d82 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()

    def forward(self, x252):
        x253=self.conv2d82(x252)
        x254=self.sigmoid16(x253)
        return x254

m = M().eval()
x252 = torch.randn(torch.Size([1, 28, 1, 1]))
start = time.time()
output = m(x252)
end = time.time()
print(end-start)
