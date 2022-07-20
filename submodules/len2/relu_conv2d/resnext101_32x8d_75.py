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
        self.relu76 = ReLU(inplace=True)
        self.conv2d80 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)

    def forward(self, x262):
        x263=self.relu76(x262)
        x264=self.conv2d80(x263)
        return x264

m = M().eval()
x262 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x262)
end = time.time()
print(end-start)