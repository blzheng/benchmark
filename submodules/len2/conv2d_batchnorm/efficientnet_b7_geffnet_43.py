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
        self.conv2d73 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d43 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x220):
        x221=self.conv2d73(x220)
        x222=self.batchnorm2d43(x221)
        return x222

m = M().eval()
x220 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x220)
end = time.time()
print(end-start)
