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
        self.relu13 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192, bias=False)

    def forward(self, x63):
        x64=self.relu13(x63)
        x65=self.conv2d20(x64)
        return x65

m = M().eval()
x63 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x63)
end = time.time()
print(end-start)
