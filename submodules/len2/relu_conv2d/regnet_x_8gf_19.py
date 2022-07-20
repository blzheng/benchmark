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
        self.relu19 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2, bias=False)

    def forward(self, x69):
        x70=self.relu19(x69)
        x71=self.conv2d22(x70)
        return x71

m = M().eval()
x69 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x69)
end = time.time()
print(end-start)
