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
        self.conv2d149 = Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)

    def forward(self, x444):
        x445=self.conv2d149(x444)
        return x445

m = M().eval()
x444 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x444)
end = time.time()
print(end-start)