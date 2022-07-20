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
        self.conv2d31 = Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x116):
        x117=torch.nn.functional.relu(x116,inplace=True)
        x118=self.conv2d31(x117)
        return x118

m = M().eval()
x116 = torch.randn(torch.Size([1, 24, 14, 14]))
start = time.time()
output = m(x116)
end = time.time()
print(end-start)
