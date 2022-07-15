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
        self.conv2d238 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)

    def forward(self, x750):
        x751=self.conv2d238(x750)
        return x751

m = M().eval()
x750 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x750)
end = time.time()
print(end-start)
