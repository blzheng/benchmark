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
        self.conv2d130 = Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)

    def forward(self, x404):
        x405=self.conv2d130(x404)
        return x405

m = M().eval()
x404 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x404)
end = time.time()
print(end-start)
