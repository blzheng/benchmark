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
        self.conv2d155 = Conv2d(2688, 2688, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2688, bias=False)

    def forward(self, x460):
        x461=self.conv2d155(x460)
        return x461

m = M().eval()
x460 = torch.randn(torch.Size([1, 2688, 7, 7]))
start = time.time()
output = m(x460)
end = time.time()
print(end-start)
