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
        self.relu35 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104, bias=False)

    def forward(self, x170):
        x171=self.relu35(x170)
        x172=self.conv2d53(x171)
        return x172

m = M().eval()
x170 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x170)
end = time.time()
print(end-start)
