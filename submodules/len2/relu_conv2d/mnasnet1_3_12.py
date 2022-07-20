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
        self.relu12 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(168, 168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=168, bias=False)

    def forward(self, x53):
        x54=self.relu12(x53)
        x55=self.conv2d19(x54)
        return x55

m = M().eval()
x53 = torch.randn(torch.Size([1, 168, 28, 28]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
