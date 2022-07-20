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
        self.conv2d192 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x599, x594):
        x600=operator.mul(x599, x594)
        x601=self.conv2d192(x600)
        return x601

m = M().eval()
x599 = torch.randn(torch.Size([1, 3072, 1, 1]))
x594 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x599, x594)
end = time.time()
print(end-start)
