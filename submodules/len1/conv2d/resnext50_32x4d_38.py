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
        self.conv2d38 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)

    def forward(self, x123):
        x124=self.conv2d38(x123)
        return x124

m = M().eval()
x123 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x123)
end = time.time()
print(end-start)
