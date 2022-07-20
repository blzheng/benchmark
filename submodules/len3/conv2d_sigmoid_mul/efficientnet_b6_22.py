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
        self.conv2d111 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()

    def forward(self, x347, x344):
        x348=self.conv2d111(x347)
        x349=self.sigmoid22(x348)
        x350=operator.mul(x349, x344)
        return x350

m = M().eval()
x347 = torch.randn(torch.Size([1, 36, 1, 1]))
x344 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x347, x344)
end = time.time()
print(end-start)
