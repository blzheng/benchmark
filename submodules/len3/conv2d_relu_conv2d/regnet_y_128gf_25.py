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
        self.conv2d131 = Conv2d(2904, 726, kernel_size=(1, 1), stride=(1, 1))
        self.relu103 = ReLU()
        self.conv2d132 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x416):
        x417=self.conv2d131(x416)
        x418=self.relu103(x417)
        x419=self.conv2d132(x418)
        return x419

m = M().eval()
x416 = torch.randn(torch.Size([1, 2904, 1, 1]))
start = time.time()
output = m(x416)
end = time.time()
print(end-start)
