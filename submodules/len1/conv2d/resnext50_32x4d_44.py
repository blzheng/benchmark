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
        self.conv2d44 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)

    def forward(self, x143):
        x144=self.conv2d44(x143)
        return x144

m = M().eval()
x143 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x143)
end = time.time()
print(end-start)
