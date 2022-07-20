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
        self.maxpool2d3 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d8 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x27):
        x28=self.maxpool2d3(x27)
        x29=self.conv2d8(x28)
        return x29

m = M().eval()
x27 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)
