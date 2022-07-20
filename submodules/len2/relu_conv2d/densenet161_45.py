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
        self.relu46 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x166):
        x167=self.relu46(x166)
        x168=self.conv2d46(x167)
        return x168

m = M().eval()
x166 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x166)
end = time.time()
print(end-start)
