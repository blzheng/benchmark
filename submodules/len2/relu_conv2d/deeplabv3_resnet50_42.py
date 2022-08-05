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
        self.relu43 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)

    def forward(self, x156):
        x157=self.relu43(x156)
        x158=self.conv2d48(x157)
        return x158

m = M().eval()
x156 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
