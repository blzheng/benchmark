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
        self.relu29 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1392, 1392, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=6, bias=False)

    def forward(self, x123):
        x124=self.relu29(x123)
        x125=self.conv2d40(x124)
        return x125

m = M().eval()
x123 = torch.randn(torch.Size([1, 1392, 28, 28]))
start = time.time()
output = m(x123)
end = time.time()
print(end-start)