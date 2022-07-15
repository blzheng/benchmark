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
        self.conv2d55 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)

    def forward(self, x176):
        x177=self.conv2d55(x176)
        return x177

m = M().eval()
x176 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x176)
end = time.time()
print(end-start)
