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
        self.conv2d69 = Conv2d(2048, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)

    def forward(self, x224):
        x225=self.conv2d69(x224)
        return x225

m = M().eval()
x224 = torch.randn(torch.Size([1, 2048, 14, 14]))
start = time.time()
output = m(x224)
end = time.time()
print(end-start)
