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
        self.conv2d76 = Conv2d(48, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()

    def forward(self, x252, x249):
        x253=self.conv2d76(x252)
        x254=self.sigmoid8(x253)
        x255=operator.mul(x254, x249)
        return x255

m = M().eval()
x252 = torch.randn(torch.Size([1, 48, 1, 1]))
x249 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x252, x249)
end = time.time()
print(end-start)