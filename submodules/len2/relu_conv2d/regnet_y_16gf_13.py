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
        self.relu17 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(448, 448, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)

    def forward(self, x73):
        x74=self.relu17(x73)
        x75=self.conv2d24(x74)
        return x75

m = M().eval()
x73 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x73)
end = time.time()
print(end-start)
