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
        self.conv2d47 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()

    def forward(self, x146, x143):
        x147=self.conv2d47(x146)
        x148=self.sigmoid8(x147)
        x149=operator.mul(x148, x143)
        return x149

m = M().eval()
x146 = torch.randn(torch.Size([1, 80, 1, 1]))
x143 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x146, x143)
end = time.time()
print(end-start)
