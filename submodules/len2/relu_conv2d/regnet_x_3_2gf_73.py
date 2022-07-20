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
        self.relu73 = ReLU(inplace=True)
        self.conv2d78 = Conv2d(1008, 1008, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=21, bias=False)

    def forward(self, x253):
        x254=self.relu73(x253)
        x255=self.conv2d78(x254)
        return x255

m = M().eval()
x253 = torch.randn(torch.Size([1, 1008, 7, 7]))
start = time.time()
output = m(x253)
end = time.time()
print(end-start)
