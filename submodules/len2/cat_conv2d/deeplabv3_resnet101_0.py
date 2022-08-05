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
        self.conv2d109 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x347, x350, x353, x356, x363):
        x364=torch.cat([x347, x350, x353, x356, x363],dim=1)
        x365=self.conv2d109(x364)
        return x365

m = M().eval()
x347 = torch.randn(torch.Size([1, 256, 28, 28]))
x350 = torch.randn(torch.Size([1, 256, 28, 28]))
x353 = torch.randn(torch.Size([1, 256, 28, 28]))
x356 = torch.randn(torch.Size([1, 256, 28, 28]))
x363 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x347, x350, x353, x356, x363)
end = time.time()
print(end-start)
