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
        self.conv2d98 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x320, x305):
        x321=operator.add(x320, x305)
        x322=self.conv2d98(x321)
        return x322

m = M().eval()
x320 = torch.randn(torch.Size([1, 224, 14, 14]))
x305 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x320, x305)
end = time.time()
print(end-start)
