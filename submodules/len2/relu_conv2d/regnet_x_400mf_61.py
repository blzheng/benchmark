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
        self.relu61 = ReLU(inplace=True)
        self.conv2d66 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)

    def forward(self, x213):
        x214=self.relu61(x213)
        x215=self.conv2d66(x214)
        return x215

m = M().eval()
x213 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x213)
end = time.time()
print(end-start)
