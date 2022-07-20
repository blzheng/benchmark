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
        self.relu17 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)

    def forward(self, x167):
        x168=self.relu17(x167)
        x169=self.conv2d27(x168)
        return x169

m = M().eval()
x167 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x167)
end = time.time()
print(end-start)
