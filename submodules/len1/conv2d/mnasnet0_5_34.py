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
        self.conv2d34 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)

    def forward(self, x97):
        x98=self.conv2d34(x97)
        return x98

m = M().eval()
x97 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x97)
end = time.time()
print(end-start)