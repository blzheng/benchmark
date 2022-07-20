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
        self.relu196 = ReLU(inplace=True)
        self.conv2d196 = Conv2d(1856, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x693):
        x694=self.relu196(x693)
        x695=self.conv2d196(x694)
        return x695

m = M().eval()
x693 = torch.randn(torch.Size([1, 1856, 7, 7]))
start = time.time()
output = m(x693)
end = time.time()
print(end-start)
