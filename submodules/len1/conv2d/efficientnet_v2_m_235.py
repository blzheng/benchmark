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
        self.conv2d235 = Conv2d(3072, 3072, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3072, bias=False)

    def forward(self, x751):
        x752=self.conv2d235(x751)
        return x752

m = M().eval()
x751 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x751)
end = time.time()
print(end-start)
