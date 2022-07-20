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
        self.relu117 = ReLU(inplace=True)
        self.conv2d117 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x416):
        x417=self.relu117(x416)
        x418=self.conv2d117(x417)
        return x418

m = M().eval()
x416 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x416)
end = time.time()
print(end-start)
