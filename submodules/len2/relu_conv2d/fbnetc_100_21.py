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
        self.relu41 = ReLU(inplace=True)
        self.conv2d62 = Conv2d(1104, 1104, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1104, bias=False)

    def forward(self, x200):
        x201=self.relu41(x200)
        x202=self.conv2d62(x201)
        return x202

m = M().eval()
x200 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x200)
end = time.time()
print(end-start)
