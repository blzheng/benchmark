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
        self.conv2d60 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)

    def forward(self, x184):
        x185=self.conv2d60(x184)
        return x185

m = M().eval()
x184 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x184)
end = time.time()
print(end-start)
