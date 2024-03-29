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
        self.relu123 = ReLU(inplace=True)
        self.conv2d123 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x437):
        x438=self.relu123(x437)
        x439=self.conv2d123(x438)
        return x439

m = M().eval()
x437 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x437)
end = time.time()
print(end-start)
