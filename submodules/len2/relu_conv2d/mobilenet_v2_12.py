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
        self.relu612 = ReLU6(inplace=True)
        self.conv2d19 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)

    def forward(self, x53):
        x54=self.relu612(x53)
        x55=self.conv2d19(x54)
        return x55

m = M().eval()
x53 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
