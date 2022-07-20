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
        self.relu25 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x95):
        x96=self.relu25(x95)
        x97=self.conv2d30(x96)
        return x97

m = M().eval()
x95 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x95)
end = time.time()
print(end-start)
