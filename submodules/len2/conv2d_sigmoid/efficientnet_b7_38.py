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
        self.conv2d190 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid38 = Sigmoid()

    def forward(self, x598):
        x599=self.conv2d190(x598)
        x600=self.sigmoid38(x599)
        return x600

m = M().eval()
x598 = torch.randn(torch.Size([1, 56, 1, 1]))
start = time.time()
output = m(x598)
end = time.time()
print(end-start)
