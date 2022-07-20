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
        self.conv2d68 = Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)

    def forward(self, x232):
        x233=torch.nn.functional.relu(x232,inplace=True)
        x234=self.conv2d68(x233)
        return x234

m = M().eval()
x232 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x232)
end = time.time()
print(end-start)
