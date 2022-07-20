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
        self.relu34 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x120):
        x121=self.relu34(x120)
        x122=self.conv2d37(x121)
        return x122

m = M().eval()
x120 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x120)
end = time.time()
print(end-start)
