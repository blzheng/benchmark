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
        self.conv2d245 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x733):
        x734=self.conv2d245(x733)
        return x734

m = M().eval()
x733 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x733)
end = time.time()
print(end-start)
