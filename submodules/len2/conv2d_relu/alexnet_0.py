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
        self.conv2d0 = Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.relu0 = ReLU(inplace=True)

    def forward(self, x0):
        x1=self.conv2d0(x0)
        x2=self.relu0(x1)
        return x2

m = M().eval()
x0 = torch.randn(torch.Size([1, 3, 224, 224]))
start = time.time()
output = m(x0)
end = time.time()
print(end-start)
