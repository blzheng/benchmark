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
        self.conv2d136 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid27 = Sigmoid()

    def forward(self, x425):
        x426=self.conv2d136(x425)
        x427=self.sigmoid27(x426)
        return x427

m = M().eval()
x425 = torch.randn(torch.Size([1, 50, 1, 1]))
start = time.time()
output = m(x425)
end = time.time()
print(end-start)
