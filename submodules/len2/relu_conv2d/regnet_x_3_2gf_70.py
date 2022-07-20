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
        self.relu70 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(1008, 1008, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=21, bias=False)

    def forward(self, x243):
        x244=self.relu70(x243)
        x245=self.conv2d75(x244)
        return x245

m = M().eval()
x243 = torch.randn(torch.Size([1, 1008, 14, 14]))
start = time.time()
output = m(x243)
end = time.time()
print(end-start)
