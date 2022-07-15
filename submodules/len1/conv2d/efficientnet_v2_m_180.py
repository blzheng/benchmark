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
        self.conv2d180 = Conv2d(1824, 1824, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1824, bias=False)

    def forward(self, x577):
        x578=self.conv2d180(x577)
        return x578

m = M().eval()
x577 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x577)
end = time.time()
print(end-start)
