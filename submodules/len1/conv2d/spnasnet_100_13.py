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
        self.conv2d13 = Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)

    def forward(self, x42):
        x43=self.conv2d13(x42)
        return x43

m = M().eval()
x42 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x42)
end = time.time()
print(end-start)
