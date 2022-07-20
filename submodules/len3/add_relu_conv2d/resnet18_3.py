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
        self.relu7 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x32, x27):
        x33=operator.add(x32, x27)
        x34=self.relu7(x33)
        x35=self.conv2d10(x34)
        return x35

m = M().eval()
x32 = torch.randn(torch.Size([1, 128, 28, 28]))
x27 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x32, x27)
end = time.time()
print(end-start)
