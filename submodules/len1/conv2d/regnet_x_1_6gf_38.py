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
        self.conv2d38 = Conv2d(408, 408, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=17, bias=False)

    def forward(self, x122):
        x123=self.conv2d38(x122)
        return x123

m = M().eval()
x122 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x122)
end = time.time()
print(end-start)
