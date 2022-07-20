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
        self.relu84 = ReLU(inplace=True)
        self.conv2d84 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x299):
        x300=self.relu84(x299)
        x301=self.conv2d84(x300)
        return x301

m = M().eval()
x299 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x299)
end = time.time()
print(end-start)
