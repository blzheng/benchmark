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
        self.relu70 = ReLU(inplace=True)
        self.conv2d76 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x251):
        x252=self.relu70(x251)
        x253=self.conv2d76(x252)
        return x253

m = M().eval()
x251 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x251)
end = time.time()
print(end-start)