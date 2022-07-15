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
        self.conv2d214 = Conv2d(3456, 3456, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3456, bias=False)

    def forward(self, x671):
        x672=self.conv2d214(x671)
        return x672

m = M().eval()
x671 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x671)
end = time.time()
print(end-start)
