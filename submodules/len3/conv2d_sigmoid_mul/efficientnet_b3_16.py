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
        self.conv2d82 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()

    def forward(self, x252, x249):
        x253=self.conv2d82(x252)
        x254=self.sigmoid16(x253)
        x255=operator.mul(x254, x249)
        return x255

m = M().eval()
x252 = torch.randn(torch.Size([1, 34, 1, 1]))
x249 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x252, x249)
end = time.time()
print(end-start)
