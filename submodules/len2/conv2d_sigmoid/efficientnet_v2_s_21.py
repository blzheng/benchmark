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
        self.conv2d127 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()

    def forward(self, x404):
        x405=self.conv2d127(x404)
        x406=self.sigmoid21(x405)
        return x406

m = M().eval()
x404 = torch.randn(torch.Size([1, 64, 1, 1]))
start = time.time()
output = m(x404)
end = time.time()
print(end-start)
