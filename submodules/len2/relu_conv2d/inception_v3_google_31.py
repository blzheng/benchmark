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
        self.conv2d58 = Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)

    def forward(self, x200):
        x201=torch.nn.functional.relu(x200,inplace=True)
        x202=self.conv2d58(x201)
        return x202

m = M().eval()
x200 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x200)
end = time.time()
print(end-start)
