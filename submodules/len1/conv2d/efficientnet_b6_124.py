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
        self.conv2d124 = Conv2d(1200, 1200, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1200, bias=False)

    def forward(self, x387):
        x388=self.conv2d124(x387)
        return x388

m = M().eval()
x387 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x387)
end = time.time()
print(end-start)
