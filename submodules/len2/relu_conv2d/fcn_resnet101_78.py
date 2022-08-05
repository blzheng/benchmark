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
        self.relu79 = ReLU(inplace=True)
        self.conv2d83 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x274):
        x275=self.relu79(x274)
        x276=self.conv2d83(x275)
        return x276

m = M().eval()
x274 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
