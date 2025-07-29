import torch.nn.functional as F
import torch
from math import exp
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def RMI_vi (input_vi,fused_result):
    RMI_vi=RMI(input_vi,fused_result)

    return RMI_vi
def RMI_ir (input_ir,fused_result ):
    RMI_ir=RMI(input_ir,fused_result)

    return RMI_ir

def ssim_vi (fused_result,input_vi ):
    ssim_vi=ssim(fused_result,input_vi)

    return ssim_vi

def ssim_ir(fused_result,input_ir):
    ssim_ir=ssim(fused_result,input_ir)

    return ssim_ir


def ssim_loss (fused_result,input_ir,input_vi ):
    ssim_loss=ssim(fused_result,torch.maximum(input_ir,input_vi))

    return ssim_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map.mean()

    return 1-ret

EPSILON = 0.0005

def RMI(x,y):
    RMI =RMILoss().to(device)
    RMI = RMI(x,y)

    return RMI

class RMILoss(nn.Module):

    def __init__(self,
                 with_logits=True,
                 radius=3,
                 bce_weight=0.5,
                 downsampling_method='max',
                 stride=3,
                 use_log_trace=True,
                 use_double_precision=True,
                 epsilon=EPSILON):

        super().__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, input, target):

        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(input, target=target)
            else:
                bce = F.binary_cross_entropy(input, target=target)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        if self.with_logits:
            input = torch.sigmoid(input)

        rmi = self.rmi_loss(input=input, target=target)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

    def rmi_loss(self, input, target):


        assert input.shape == target.shape
        vector_size = self.radius * self.radius

        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input)

        if self.use_double_precision:
            y = y.double()
            p = p.double()

        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        y_cov = y @ transpose(y)
        p_cov = p @ transpose(p)
        y_p_cov = y @ transpose(p)

        m = y_cov - y_p_cov @ transpose(inverse(p_cov + eps)) @ transpose(y_p_cov)

        if self.use_log_trace:
            rmi = 0.5 * log_trace(m + eps)
        else:
            rmi = 0.5 * log_det(m + eps)

        rmi = rmi / float(vector_size)

        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):


        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == 'region-extraction' else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius ** 2, -1))
        return x_regions

    def downsample(self, x):

        if self.stride == 1:
            return x

        if self.downsampling_method == 'region-extraction':
            return x

        padding = self.stride // 2
        if self.downsampling_method == 'max':
            return F.max_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        if self.downsampling_method == 'avg':
            return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        raise ValueError(self.downsampling_method)


def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x):
    # x = torch.cholesky(x)
    x = torch.linalg.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def log_det(x):
    return torch.logdet(x)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_int(fusion_image, images_ir, images_vi):
    g_loss_int = torch.mean(torch.square(fusion_image - images_ir)) + \
                5*torch.mean(torch.square(fusion_image - images_vi))

    return g_loss_int


def loss_grad(fusion_image, images_ir, images_vi):
    Image_vi_grad_lowpass = torch.abs(gradient(low_pass(images_vi)))
    Image_ir_grad_lowpass = torch.abs(gradient(low_pass(images_ir)))

    Image_vi_weight_lowpass = Image_vi_grad_lowpass
    Image_ir_weight_lowpass = Image_ir_grad_lowpass
    Image_vi_score_1 = 1

    Image_vi_score_2 = torch.sign(
        Image_vi_weight_lowpass - torch.minimum(Image_vi_weight_lowpass, Image_ir_weight_lowpass))

    Image_vi_score = Image_vi_score_1 * Image_vi_score_2
    Image_ir_score = 1 - Image_vi_score
    g_loss_grad = torch.mean(Image_ir_score * torch.square(gradient(fusion_image) - gradient(images_ir))) + \
                  torch.mean(Image_vi_score * torch.square(gradient(fusion_image) - gradient(images_vi)))

    return g_loss_grad


def gradient(image):
    _, _, h, w = image.shape
    # print(h, w)
    img = image
    k = torch.tensor([0., 1., 0., 1., -4., 1., 0., 1., 0.], dtype=torch.float)
    k = k.view(1, 1, 3, 3).to(device)
    # print(k.size(), k)
    z = F.conv2d(img, k, padding=1)
    result = z
    # print(result.shape)
    return result


def low_pass(image):
    _, _, h, w = image.shape
    # print(h, w)
    img = image
    k = torch.tensor([0.0947, 0.1183, 0.0947, 0.1183, 0.1478, 0.1183, 0.0947, 0.1183, 0.0947], dtype=torch.float)
    k = k.view(1, 1, 3, 3).to(device)
    # print(k.size(), k)
    z = F.conv2d(img, k, padding=1)
    result = z
    # print(result.shape)
    return result


def loss_de(sept_ir, sept_vi, images_ir, images_vi):
    g_loss_sept = torch.mean(torch.square(sept_ir - images_ir)) + torch.mean(torch.square(sept_vi - images_vi))
    return g_loss_sept


def loss_total(fusion_image, images_ir, images_vi):
    g_loss = 10 * loss_int(fusion_image, images_ir, images_vi) + \
             1 * loss_grad(fusion_image, images_ir, images_vi)
    return g_loss



if __name__ == '__main__':
    x1 = torch.randn(1, 1, 256, 256).to(device)
    x2 = torch.randn(1, 1, 256, 256).to()
    x3 = torch.randn(1, 1, 256, 256).to(device)
    x4 = torch.randn(1, 1, 256, 256).to(device)
    x5 = torch.randn(1, 1, 256, 256).to(device)
    # z = loss_de(x1, x2, x3, x4)
    # print(z)

    z = loss_total(x1, x2, x3, x4, x5)
    print(z)
    # Test the loss function with random inputs

    # Define a random tensor for fused images and source images
    batch_size = 4
    height, width = 256, 256

    # Random fused images and source images (simulating MRI and SPECT)
    fusion_image = torch.rand(batch_size, height, width)  # Simulated fused image
    images_ir = torch.rand(batch_size, height, width)  # Simulated MRI image
    images_vi = torch.rand(batch_size, height, width)  # Simulated SPECT image

