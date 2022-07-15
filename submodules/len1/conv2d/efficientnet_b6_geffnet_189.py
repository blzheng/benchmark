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
        self.conv2d189 = Conv2d(2064, 2064, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2064, bias=False)

    def forward(self, x564):
        x565=self.conv2d189(x564)
        return x565

m = M().eval()
x564 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x564)
end = time.time()
print(end-start)
