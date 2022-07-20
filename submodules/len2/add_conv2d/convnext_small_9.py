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
        self.conv2d15 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)

    def forward(self, x149, x139):
        x150=operator.add(x149, x139)
        x152=self.conv2d15(x150)
        return x152

m = M().eval()
x149 = torch.randn(torch.Size([1, 384, 14, 14]))
x139 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x149, x139)
end = time.time()
print(end-start)
