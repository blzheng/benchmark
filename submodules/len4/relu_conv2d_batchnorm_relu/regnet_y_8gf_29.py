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
        self.relu65 = ReLU(inplace=True)
        self.conv2d86 = Conv2d(2016, 2016, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=36, bias=False)
        self.batchnorm2d54 = BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU(inplace=True)

    def forward(self, x269):
        x270=self.relu65(x269)
        x271=self.conv2d86(x270)
        x272=self.batchnorm2d54(x271)
        x273=self.relu66(x272)
        return x273

m = M().eval()
x269 = torch.randn(torch.Size([1, 2016, 14, 14]))
start = time.time()
output = m(x269)
end = time.time()
print(end-start)
