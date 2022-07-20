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
        self.relu126 = ReLU(inplace=True)
        self.conv2d126 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x446):
        x447=self.relu126(x446)
        x448=self.conv2d126(x447)
        return x448

m = M().eval()
x446 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x446)
end = time.time()
print(end-start)
