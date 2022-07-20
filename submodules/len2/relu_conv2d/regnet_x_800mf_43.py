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
        self.relu43 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=42, bias=False)

    def forward(self, x153):
        x154=self.relu43(x153)
        x155=self.conv2d48(x154)
        return x155

m = M().eval()
x153 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
