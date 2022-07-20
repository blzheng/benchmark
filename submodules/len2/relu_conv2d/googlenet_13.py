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
        self.conv2d41 = Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x150):
        x151=torch.nn.functional.relu(x150,inplace=True)
        x152=self.conv2d41(x151)
        return x152

m = M().eval()
x150 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x150)
end = time.time()
print(end-start)
