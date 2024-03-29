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
        self.relu145 = ReLU(inplace=True)
        self.conv2d152 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x501):
        x502=self.relu145(x501)
        x503=self.conv2d152(x502)
        return x503

m = M().eval()
x501 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x501)
end = time.time()
print(end-start)
