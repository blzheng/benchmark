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
        self.conv2d234 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x701):
        x702=self.conv2d234(x701)
        return x702

m = M().eval()
x701 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x701)
end = time.time()
print(end-start)
