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
        self.conv2d47 = Conv2d(40, 640, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()

    def forward(self, x155):
        x156=self.conv2d47(x155)
        x157=self.sigmoid4(x156)
        return x157

m = M().eval()
x155 = torch.randn(torch.Size([1, 40, 1, 1]))
start = time.time()
output = m(x155)
end = time.time()
print(end-start)