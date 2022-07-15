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
        self.conv2d44 = Conv2d(352, 352, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=352, bias=False)

    def forward(self, x280):
        x281=self.conv2d44(x280)
        return x281

m = M().eval()
x280 = torch.randn(torch.Size([1, 352, 14, 14]))
start = time.time()
output = m(x280)
end = time.time()
print(end-start)
