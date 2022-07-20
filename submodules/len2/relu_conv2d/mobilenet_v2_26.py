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
        self.relu626 = ReLU6(inplace=True)
        self.conv2d40 = Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)

    def forward(self, x114):
        x115=self.relu626(x114)
        x116=self.conv2d40(x115)
        return x116

m = M().eval()
x114 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x114)
end = time.time()
print(end-start)
