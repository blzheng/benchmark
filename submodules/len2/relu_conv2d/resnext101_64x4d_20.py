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
        self.relu19 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x77):
        x78=self.relu19(x77)
        x79=self.conv2d24(x78)
        return x79

m = M().eval()
x77 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x77)
end = time.time()
print(end-start)
