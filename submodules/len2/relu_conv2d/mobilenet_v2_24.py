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
        self.relu624 = ReLU6(inplace=True)
        self.conv2d37 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)

    def forward(self, x105):
        x106=self.relu624(x105)
        x107=self.conv2d37(x106)
        return x107

m = M().eval()
x105 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
