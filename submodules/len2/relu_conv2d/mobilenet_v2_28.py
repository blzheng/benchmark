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
        self.relu628 = ReLU6(inplace=True)
        self.conv2d43 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)

    def forward(self, x122):
        x123=self.relu628(x122)
        x124=self.conv2d43(x123)
        return x124

m = M().eval()
x122 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x122)
end = time.time()
print(end-start)
