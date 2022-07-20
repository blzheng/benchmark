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
        self.relu46 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)

    def forward(self, x161):
        x162=self.relu46(x161)
        x163=self.conv2d50(x162)
        return x163

m = M().eval()
x161 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x161)
end = time.time()
print(end-start)