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
        self.relu45 = ReLU(inplace=True)
        self.conv2d60 = Conv2d(896, 896, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)

    def forward(self, x187):
        x188=self.relu45(x187)
        x189=self.conv2d60(x188)
        return x189

m = M().eval()
x187 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x187)
end = time.time()
print(end-start)
