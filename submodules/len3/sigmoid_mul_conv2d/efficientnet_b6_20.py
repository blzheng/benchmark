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
        self.sigmoid20 = Sigmoid()
        self.conv2d102 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x316, x312):
        x317=self.sigmoid20(x316)
        x318=operator.mul(x317, x312)
        x319=self.conv2d102(x318)
        return x319

m = M().eval()
x316 = torch.randn(torch.Size([1, 864, 1, 1]))
x312 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x316, x312)
end = time.time()
print(end-start)
