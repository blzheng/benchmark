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
        self.conv2d179 = Conv2d(1344, 1344, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1344, bias=False)

    def forward(self, x580):
        x581=self.conv2d179(x580)
        return x581

m = M().eval()
x580 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x580)
end = time.time()
print(end-start)
