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
        self.conv2d12 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x39):
        x40=self.relu11(x39)
        x41=self.conv2d12(x40)
        return x41

m = M().eval()
x39 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x39)
end = time.time()
print(end-start)
