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
        self.conv2d269 = Conv2d(2304, 2304, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2304, bias=False)

    def forward(self, x866):
        x867=self.conv2d269(x866)
        return x867

m = M().eval()
x866 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x866)
end = time.time()
print(end-start)
