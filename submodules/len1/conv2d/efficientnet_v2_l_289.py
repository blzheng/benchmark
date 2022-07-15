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
        self.conv2d289 = Conv2d(2304, 2304, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2304, bias=False)

    def forward(self, x930):
        x931=self.conv2d289(x930)
        return x931

m = M().eval()
x930 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x930)
end = time.time()
print(end-start)
