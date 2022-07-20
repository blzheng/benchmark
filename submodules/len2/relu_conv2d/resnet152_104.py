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
        self.relu103 = ReLU(inplace=True)
        self.conv2d109 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x359):
        x360=self.relu103(x359)
        x361=self.conv2d109(x360)
        return x361

m = M().eval()
x359 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x359)
end = time.time()
print(end-start)
