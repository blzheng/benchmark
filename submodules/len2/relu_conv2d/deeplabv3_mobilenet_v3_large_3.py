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
        self.relu3 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)

    def forward(self, x21):
        x22=self.relu3(x21)
        x23=self.conv2d7(x22)
        return x23

m = M().eval()
x21 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
