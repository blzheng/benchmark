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
        self.conv2d206 = Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x661):
        x662=self.conv2d206(x661)
        return x662

m = M().eval()
x661 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x661)
end = time.time()
print(end-start)