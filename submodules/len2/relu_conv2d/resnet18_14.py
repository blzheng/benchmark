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
        self.relu15 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x61):
        x62=self.relu15(x61)
        x63=self.conv2d19(x62)
        return x63

m = M().eval()
x61 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)
