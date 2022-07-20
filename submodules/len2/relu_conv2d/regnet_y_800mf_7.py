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
        self.relu9 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=9, bias=False)

    def forward(self, x41):
        x42=self.relu9(x41)
        x43=self.conv2d14(x42)
        return x43

m = M().eval()
x41 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x41)
end = time.time()
print(end-start)
