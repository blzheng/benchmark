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
        self.conv2d31 = Conv2d(112, 448, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x96):
        x97=self.conv2d31(x96)
        return x97

m = M().eval()
x96 = torch.randn(torch.Size([1, 112, 1, 1]))
start = time.time()
output = m(x96)
end = time.time()
print(end-start)
