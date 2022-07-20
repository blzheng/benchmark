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
        self.conv2d2 = Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x19):
        x20=torch.nn.functional.relu(x19,inplace=True)
        x21=self.conv2d2(x20)
        return x21

m = M().eval()
x19 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)
