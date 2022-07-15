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
        self.conv2d150 = Conv2d(1632, 1632, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1632, bias=False)

    def forward(self, x446):
        x447=self.conv2d150(x446)
        return x447

m = M().eval()
x446 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x446)
end = time.time()
print(end-start)
