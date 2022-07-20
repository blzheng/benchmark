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
        self.relu22 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(720, 720, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=6, bias=False)

    def forward(self, x81):
        x82=self.relu22(x81)
        x83=self.conv2d26(x82)
        return x83

m = M().eval()
x81 = torch.randn(torch.Size([1, 720, 28, 28]))
start = time.time()
output = m(x81)
end = time.time()
print(end-start)
