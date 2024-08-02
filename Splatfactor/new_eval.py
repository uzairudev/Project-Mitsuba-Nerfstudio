import torch
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt

#torch.save(tensor, 'first_sample.pth')
#torch.save(tensor, 'second_sample.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

first_sample = torch.load('first_sample.pth', weights_only=True).to(device)
second_sample = torch.load('second_sample.pth', weights_only=True).to(device)
# second_sample = torch.load('first_sample.pth', weights_only=True).to(device)

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = SSIM(data_range=1.0, size_average=True, channel=3)
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

psnr_value = psnr(first_sample, second_sample)
ssim_value = ssim(first_sample, second_sample)
lpips_value = lpips(first_sample, second_sample)

metrics = ['PSNR', 'SSIM', 'LPIPS']
values = [psnr_value, ssim_value, lpips_value]

plt.title('Comparison of PSNR, SSIM, and LPIPS Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')

from pprint import pprint

for i, value in enumerate(values):
    plt.text(i, value + 0.05, f'{value:.2f}', ha='center')
    pprint(value)

plt.show()
