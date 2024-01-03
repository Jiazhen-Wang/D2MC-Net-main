from torch import nn
from utils.utils import *
from network.unet import Unet


class KU_Module(nn.Module):
    def __init__(self, channel=64):
        super(KU_Module, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.conv(x)
        return out


class D2MC_Net(nn.Module):
    def __init__(self):
        super(D2MC_Net, self).__init__()
        self.image_model_1 = Unet(2, 2, 32, 4, batchnorm=torch.nn.InstanceNorm2d, init_type='none')
        self.image_model_2 = Unet(2, 2, 32, 4, batchnorm=torch.nn.InstanceNorm2d, init_type='none')
        self.image_model_3 = Unet(2, 2, 32, 4, batchnorm=torch.nn.InstanceNorm2d, init_type='none')
        self.ku_model_1 = KU_Module()
        self.ku_model_2 = KU_Module()
        self.ku_model_3 = KU_Module()
        self.kspace_model_1 = Unet(2, 2, 32, 4, batchnorm=torch.nn.InstanceNorm2d, init_type='none')
        self.kspace_model_2 = Unet(2, 2, 32, 4, batchnorm=torch.nn.InstanceNorm2d, init_type='none')
        self.kspace_model_3 = Unet(2, 2, 32, 4, batchnorm=torch.nn.InstanceNorm2d, init_type='none')

        self.lam1 = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.lam2 = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.lam3 = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.rho1 = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.rho2 = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.rho3 = nn.Parameter(torch.Tensor([1]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        # get k-space data
        x_k = fft2(x.permute(0, 2, 3, 1))
        x_k_1 = x_k.permute(0, 3, 1, 2)

        ############   stage_1   ############
        # k-space uncertainty
        attention1 = self.ku_model_1(x_k_1)

        # update in k-space domain
        k_space =  x_k_1
        kspace_1 = self.kspace_model_1(k_space)
        kspace_1 = (attention1 * attention1 * x_k_1 + self.lam1*kspace_1)/(attention1*attention1+self.lam1)
        kspace_1_img = ifft2(kspace_1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # update in image domain
        image_1 = self.image_model_1(kspace_1_img)
        kspace = fft2(image_1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        k_space = (attention1 * attention1 * x_k_1 + self.rho1* kspace)/(attention1*attention1+self.rho1)
        kspace_1_img = ifft2(k_space.permute(0, 2, 3, 1))
        image_1 = kspace_1_img.view(b, c, h, w)

        ############   stage_2   ############
        # k-space uncertainty
        res_2 = k_space - x_k_1
        attention2 = self.ku_model_2(res_2)

        # update in k-space domain
        kspace_2 = self.kspace_model_2(k_space)
        kspace_2 = (attention2 * attention2 * x_k_1 + self.lam2*kspace_2)/(attention2*attention2+self.lam2)
        kspace_2_img = ifft2(kspace_2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # update in image domain
        image_2 = self.image_model_2(kspace_2_img)
        kspace = fft2(image_2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        k_space = (attention2 * attention2 * x_k_1 + self.rho2*kspace)/(attention2*attention2+self.rho2)
        kspace_2_img  = ifft2(k_space.permute(0, 2, 3, 1))
        image_2 = kspace_2_img.view(b, c, h, w)

        ############   stage_3   ############
        # k-space uncertainty
        res_3 = k_space - x_k_1
        attention3 = self.ku_model_3(res_3)

        # update in k-space domain
        kspace_3 = self.kspace_model_3(k_space)
        kspace_3 = (attention3 * attention3 * x_k_1 + self.lam3*kspace_3)/(attention3*attention3+self.lam3)
        kspace_3_img = ifft2(kspace_3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # update in image domain
        image_3 = self.image_model_3(kspace_3_img)
        kspace = fft2(image_3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        k_space = (attention3 * attention3 * x_k_1 + self.rho3*kspace)/(attention3*attention3+self.rho3)
        kspace_3_img = ifft2(k_space.permute(0, 2, 3, 1))
        image_3 = kspace_3_img.view(b, c, h, w)


        return image_3, image_2, image_1, kspace_3, kspace_2, kspace_1



if __name__ == '__main__':
    x = torch.randn(1, 2, 256, 256)
    net = D2MC_Net()
    rec_3, rec_2, rec_1, k_space_3, k_space_2, k_space_1 = net(x)
    print(rec_3.shape)


