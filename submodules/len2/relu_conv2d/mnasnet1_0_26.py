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
        self.relu26 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)

    def forward(self, x113):
        x114=self.relu26(x113)
        x115=self.conv2d40(x114)
        return x115

m = M().eval()
x113 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x113)
end = time.time()
print(end-start)
