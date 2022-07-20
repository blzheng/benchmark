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
        self.relu61 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x219):
        x220=self.relu61(x219)
        x221=self.conv2d67(x220)
        return x221

m = M().eval()
x219 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x219)
end = time.time()
print(end-start)
