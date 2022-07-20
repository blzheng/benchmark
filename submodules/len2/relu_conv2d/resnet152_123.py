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
        self.relu124 = ReLU(inplace=True)
        self.conv2d128 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x422):
        x423=self.relu124(x422)
        x424=self.conv2d128(x423)
        return x424

m = M().eval()
x422 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x422)
end = time.time()
print(end-start)
