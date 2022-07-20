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
        self.relu97 = ReLU(inplace=True)
        self.conv2d125 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)

    def forward(self, x395):
        x396=self.relu97(x395)
        x397=self.conv2d125(x396)
        return x397

m = M().eval()
x395 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x395)
end = time.time()
print(end-start)
