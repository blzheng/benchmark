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
        self.conv2d6 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)

    def forward(self, x18):
        x19=self.conv2d6(x18)
        return x19

m = M().eval()
x18 = torch.randn(torch.Size([1, 96, 112, 112]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
