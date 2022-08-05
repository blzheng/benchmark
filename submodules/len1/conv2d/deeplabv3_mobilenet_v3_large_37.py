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
        self.conv2d37 = Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)

    def forward(self, x110):
        x111=self.conv2d37(x110)
        return x111

m = M().eval()
x110 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x110)
end = time.time()
print(end-start)
