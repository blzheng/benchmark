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
        self.conv2d30 = Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)

    def forward(self, x89):
        x90=self.conv2d30(x89)
        return x90

m = M().eval()
x89 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)