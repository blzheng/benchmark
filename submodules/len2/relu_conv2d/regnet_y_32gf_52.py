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
        self.relu69 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(1392, 1392, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)

    def forward(self, x283):
        x284=self.relu69(x283)
        x285=self.conv2d90(x284)
        return x285

m = M().eval()
x283 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x283)
end = time.time()
print(end-start)
