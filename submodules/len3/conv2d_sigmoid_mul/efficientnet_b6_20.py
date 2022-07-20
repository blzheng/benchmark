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
        self.conv2d101 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()

    def forward(self, x315, x312):
        x316=self.conv2d101(x315)
        x317=self.sigmoid20(x316)
        x318=operator.mul(x317, x312)
        return x318

m = M().eval()
x315 = torch.randn(torch.Size([1, 36, 1, 1]))
x312 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x315, x312)
end = time.time()
print(end-start)
