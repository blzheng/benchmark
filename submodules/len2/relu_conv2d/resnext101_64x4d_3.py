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
        self.relu4 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)

    def forward(self, x18):
        x19=self.relu4(x18)
        x20=self.conv2d6(x19)
        return x20

m = M().eval()
x18 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
