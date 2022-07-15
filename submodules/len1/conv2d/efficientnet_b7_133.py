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
        self.conv2d133 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)

    def forward(self, x418):
        x419=self.conv2d133(x418)
        return x419

m = M().eval()
x418 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x418)
end = time.time()
print(end-start)
