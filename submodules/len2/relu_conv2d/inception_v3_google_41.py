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
        self.conv2d75 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), bias=False)

    def forward(self, x255):
        x256=torch.nn.functional.relu(x255,inplace=True)
        x257=self.conv2d75(x256)
        return x257

m = M().eval()
x255 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x255)
end = time.time()
print(end-start)
