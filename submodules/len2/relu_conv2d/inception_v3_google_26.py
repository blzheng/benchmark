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
        self.conv2d52 = Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)

    def forward(self, x182):
        x183=torch.nn.functional.relu(x182,inplace=True)
        x184=self.conv2d52(x183)
        return x184

m = M().eval()
x182 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
