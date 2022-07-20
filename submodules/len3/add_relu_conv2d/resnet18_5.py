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
        self.relu11 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x48, x43):
        x49=operator.add(x48, x43)
        x50=self.relu11(x49)
        x51=self.conv2d15(x50)
        return x51

m = M().eval()
x48 = torch.randn(torch.Size([1, 256, 14, 14]))
x43 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x48, x43)
end = time.time()
print(end-start)
