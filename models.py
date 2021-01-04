import torch

MIN_CHANNELS = 32


## General convBlock
class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,k_s=3,stride=1,padding=1) -> None:
        super(ConvBlock, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_s, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

### ConvBlock for discriminator, we use a bigger kernel side to watch for larger details
class ConvBlockD(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlockD, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
      


class SingleScaleGenerator(torch.nn.Module):

    def __init__(self, n_channels: int = 32, min_channels: int = MIN_CHANNELS, n_blocks=5) -> None:
        super(SingleScaleGenerator, self).__init__()

        self.head = ConvBlock(in_channels=3, out_channels=n_channels)

        self.body = torch.nn.ModuleList()
        for i in range(n_blocks-2):
            in_channels = max([min_channels, n_channels // (2 ** (i))])
            out_channels = max([min_channels, n_channels // (2 ** (i+1))])
            self.body.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
        self.body = torch.nn.Sequential(*self.body)

        self.tail = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r = x + z
        r = self.tail(self.body(self.head(r)))
        return x + r


class Discriminator(torch.nn.Module):

    def __init__(self, n_channels: int = 32, min_channels: int = MIN_CHANNELS, n_blocks=5,head_stride=1) -> None:
        super(Discriminator, self).__init__()

        self.head = ConvBlock(in_channels=3, out_channels=n_channels,k_s=11,stride=head_stride,padding=5)

        self.body = torch.nn.ModuleList()
        for i in range(n_blocks-2):
            in_channels = max([min_channels, n_channels // (2 ** (i))])
            out_channels = max([min_channels, n_channels // (2 ** (i+1))])
            self.body.append(ConvBlockD(in_channels=in_channels, out_channels=out_channels))
        self.body = torch.nn.Sequential(*self.body)

        self.tail = torch.nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=3, stride=1, padding=1)


    def forward(self, x) -> None:
        #return self.tail(self.body(self.head(x)))
        x = self.body(self.head(x))
        
        return x.reshape((x.shape[1],-1)).T

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)