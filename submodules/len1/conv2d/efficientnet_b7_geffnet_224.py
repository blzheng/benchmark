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
        self.conv2d224 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x671):
        x672=self.conv2d224(x671)
        return x672

m = M().eval()
x671 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x671)
end = time.time()
print(end-start)
