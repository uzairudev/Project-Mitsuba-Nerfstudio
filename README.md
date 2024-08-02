# Spectral Data Generation using Mitsuba and TOUCAN Multispectral Camera

This repository contains code and data for generating spectral data using the Mitsuba renderer and the TOUCAN Multispectral Camera. The project is organized into several Jupyter notebooks that handle different spectral bands, as well as encoded bands.


## Repository Structure

```
.
├── Splatfactor/
│   ├── extracted_blue.py
│   ├── extracted_green.py
│   ├── extracted_red.py
│   ├── first_sample.pth
│   ├── new_eval.py
│   ├── second_sample.pth
│   └── splatfacto.py
├── _MACOSX/lego/
├── lego/
│   ├── cbox.xml
│   ├── [Lego meshes and textures]
├── Gaussian-like-curve.ipynb
├── mitsuba_Encoded1.ipynb
├── mitsuba_Encoded2.ipynb
├── mitsuba_Encoded3.ipynb
├── mitsuba_RGB.ipynb
├── mitsuba_band1.ipynb
├── mitsuba_band2.ipynb
├── mitsuba_band3.ipynb
├── mitsuba_band4.ipynb
├── mitsuba_band5.ipynb
├── mitsuba_band6.ipynb
├── mitsuba_band7.ipynb
├── mitsuba_band8.ipynb
├── mitsuba_band9.ipynb
├── mitsubaspectral.ipynb
├── my_first_render.png
└── pexels-fwstudio-33348-172289.jpg
```

### Notebooks

- **Gaussian-like-curve.ipynb**: Plots the sensitivity of the TOUCAN Multispectral Camera for bands 1 to 9.
- **mitsuba_Encoded1.ipynb**: Generates data for encoded bands 1-3.
- **mitsuba_Encoded2.ipynb**: Generates data for encoded bands 4-6.
- **mitsuba_Encoded3.ipynb**: Generates data for encoded bands 7-9.
- **mitsuba_RGB.ipynb**: Handles RGB rendering.
- **mitsuba_band1.ipynb**: Generates data for individual spectral band 1.
- **mitsuba_band2.ipynb**: Generates data for individual spectral band 2.
- **mitsuba_band3.ipynb**: Generates data for individual spectral band 3.
- **mitsuba_band4.ipynb**: Generates data for individual spectral band 4.
- **mitsuba_band5.ipynb**: Generates data for individual spectral band 5.
- **mitsuba_band6.ipynb**: Generates data for individual spectral band 6.
- **mitsuba_band7.ipynb**: Generates data for individual spectral band 7.
- **mitsuba_band8.ipynb**: Generates data for individual spectral band 8.
- **mitsuba_band9.ipynb**: Generates data for individual spectral band 9.
- **mitsubaspectral.ipynb**: A general notebook for spectral data generation.

### Additional Files

- **Splatfactor/**:
  - **extracted_blue.py**: Extracts and processes the blue channel data.
  - **extracted_green.py**: Extracts and processes the green channel data.
  - **extracted_red.py**: Extracts and processes the red channel data.
  - **first_sample.pth**: PyTorch tensor file containing the first sample.
  - **new_eval.py**: Evaluation script to compare two samples using PSNR, SSIM, and LPIPS metrics.
  - **second_sample.pth**: PyTorch tensor file containing the second sample.
  - **splatfacto.py**: Contains the implementation of the `SplatfactoModel` for Gaussian Splatting.
- **lego/**: Contains the `cbox.xml` scene file, along with all necessary Lego meshes and textures for rendering.
- **pexels-fwstudio-33348-172289.jpg**: Image used as a background texture for the LEGO model.

### `extracted_blue.py`, `extracted_green.py`, `extracted_red.py`
These scripts extract and process the color channel data for blue, green, and red, respectively. They are used to analyze and manipulate the individual color channels separately.

### `first_sample.pth` and `second_sample.pth`
These files contain PyTorch tensor data for the first and second samples used for evaluation. They are loaded and compared using the `new_eval.py` script.

## Spectral Sensitivity Curve
The spectral sensitivity of the TOUCAN Multispectral Camera is defined by a set of Gaussian functions representing different spectral bands. The following Python code in `Gaussian-like-curve.ipynb` generates and plots these sensitivity curves:

```python
import matplotlib.pyplot as plt
import numpy as np

def gaussian(x, mu, sigma, max_value):
    return max_value * np.exp(-0.5 * ((x - mu) / sigma)**2)

bands = [
    {"center": 431, "fwhm": 65, "max_t": 0.36},
    {"center": 479, "fwhm": 47, "max_t": 0.45},
    {"center": 515, "fwhm": 42, "max_t": 0.49},
    {"center": 567, "fwhm": 37, "max_t": 0.54},
    {"center": 611, "fwhm": 36, "max_t": 0.55},
    {"center": 666, "fwhm": 34, "max_t": 0.56},
    {"center": 719, "fwhm": 33, "max_t": 0.53},
    {"center": 775, "fwhm": 31, "max_t": 0.51},
    {"center": 820, "fwhm": 31, "max_t": 0.49},
]

wavelengths = np.linspace(400, 900, 1000)
sensitivity_curves = np.zeros((10, len(wavelengths)))

for i, band in enumerate(bands):
    sigma = band["fwhm"] / 2.355
    sensitivity_curves[i, :] = gaussian(wavelengths, band["center"], sigma, band["max_t"])

plt.figure(figsize=(12, 8))
for i, band in enumerate(bands):
    plt.plot(wavelengths, sensitivity_curves[i, :], label=f'Band {i+1} (λc = {band["center"]} nm, FWHM = {band["fwhm"]} nm, Tmax = {band["max_t"]*100}%)')

plt.title('Spectral Sensitivity of TOUCAN Multispectral Camera')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative Sensitivity')
plt.legend()
plt.grid(True)
plt.show()
```

## Sample Code for Data Generation for Band 1

The following sample code demonstrates how to generate data for band 1 using the Mitsuba renderer:

```python
import mitsuba as mi
mi.variants()
mi.set_variant("scalar_spectral")
scene = mi.load_file("lego/cbox.xml")

from mitsuba import ScalarTransform4f as T

def load_sensor(r, phi, theta):
    origin = T.translate([0, -1.2, 0]) @ T.rotate([0, 1, 0], phi).rotate([1, 0, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.37,
        'to_world': T.look_at(
            origin=origin,
            target=[0, -3.7, 0],
            up=[0, 1, 0]
        ),
        'sampler': {
            'type': 'independent',
        },
        'film': {
            'type': 'specfilm',
            'width': 2000,
            'height': 2000,
            'band1': {
                'type': 'spectrum',
                'value': [(400.0, 0.01), (415.0, 0.15), (431.0, 0.36), (445.0, 0.15), (460.0, 0.01)]
            }
        },
    })

radius = 2
phis = [20.0*(i+1) for i in range(10)]
thetas = [20.0*(j+1) for j in range(10)]

sensors = []
for phi in phis:
    for theta in thetas:
        sensors.append(load_sensor(radius,theta,phi))
sensor_count = len(sensors)

images = [mi.render(scene, spp=512, sensor=sensor) for sensor in sensors]

import os
folder_name = "rendered_images"

os.makedirs(folder_name, exist_ok=True)

for i, image in enumerate(images):
    filename = f"{folder_name}/view_{i+1:04d}.png"
    mi.util.write_bitmap(filename, image)
    
print(f"Rendered images saved to: {folder_name}")
```


## Sample Code For Extracted different Bands from the Model. 

### Replicate the R Channel in B and G Channels for Both Ground Truth and Predicted Images

In the `splatfacto.py` file, there is a method that replicates the R channel in the B and G channels for both ground truth and predicted images. This is done to facilitate certain image quality assessments and visualizations.

Here is the relevant function with the corresponding extraction logic:

```python
def get_image_metrics_and_images(
    self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """Writes the test image outputs.

    Args:
        image_idx: Index of the image.
        step: Current step.
        batch: Batch of data.
        outputs: Outputs of the model.

    Returns:
        A dictionary of metrics.
    """
    gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
    predicted_rgb = outputs["rgb"]

    # Replicate the R channel in B and G channels for both ground truth and predicted images
    gt_rgb_r = gt_rgb[:, :, 0:1]  # Extract R channel
    gt_rgb = torch.cat([gt_rgb_r, gt_rgb_r, gt_rgb_r], dim=2)  # Replicate R channel into G and B

    predicted_rgb_r = predicted_rgb[:, :, 0:1]  # Extract R channel
    predicted_rgb = torch.cat([predicted_rgb_r, predicted_rgb_r, predicted_rgb_r], dim=2)  # Replicate R channel into G and B

    combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

    # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
    gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[

None, ...]
    predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

    psnr = self.psnr(gt_rgb, predicted_rgb)
    ssim = self.ssim(gt_rgb, predicted_rgb)
    lpips = self.lpips(gt_rgb, predicted_rgb)

    # all of these metrics will be logged as scalars
    metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
    metrics_dict["lpips"] = float(lpips)

    images_dict = {"img": combined_rgb}

    return metrics_dict, images_dict
```

This method:
1. Extracts the red (R) channel from the ground truth and predicted images.
2. Replicates the R channel across the green (G) and blue (B) channels.
3. Combines the ground truth and predicted images side-by-side.
4. Computes PSNR, SSIM, and LPIPS metrics for the images.
5. Returns a dictionary containing the metrics and the combined image.


## Sample Code For Comparing only the predicted output as saved Tensors 
The  `new_eval.py` script is used to compare two samples (stored in `first_sample.pth` and `second_sample.pth`) using three image quality metrics: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index Measure), and LPIPS (Learned Perceptual Image Patch Similarity). The script loads the samples, computes the metrics, and visualizes the results using a bar plot.
Here is the `new_eval.py` script:

```python
import torch
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load samples
first_sample = torch.load('first_sample.pth', weights_only=True).to(device)
second_sample = torch.load('second_sample.pth', weights_only=True).to(device)

# Initialize metrics
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

# Compute metrics
psnr_value = psnr(first_sample, second_sample)
ssim_value = ssim(first_sample, second_sample)
lpips_value = lpips(first_sample, second_sample)

# Plot results
metrics = ['PSNR', 'SSIM', 'LPIPS']
values = [psnr_value.item(), ssim_value.item(), lpips_value.item()]

plt.bar(metrics, values)
plt.title('Comparison of PSNR, SSIM, and LPIPS Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')

for i, value in enumerate(values):
    plt.text(i, value + 0.05, f'{value:.2f}', ha='center')

plt.show()
```

## Usage

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. **Install required dependencies**:
   Ensure you have Jupyter Notebook and the necessary Python libraries installed (e.g., `matplotlib`, `numpy`, `mitsuba`, `torch`, `pytorch-msssim`, `torchmetrics`).

3. **Navigate through the Jupyter notebooks**:
   Open and run the Jupyter notebooks to generate the spectral data for different bands.

4. **Render the scenes**:
   Use Mitsuba to render the scenes specified in the `lego/cbox.xml` file using the spectral sensitivity settings.

5. **Run the evaluation script**:
   Use the `new_eval.py` script to compare the two samples using PSNR, SSIM, and LPIPS metrics.


## Acknowledgements

- [Mitsuba Renderer](https://www.mitsuba-renderer.org/) for providing the rendering framework.
- [Nerfstudio](https://nerfstudio.github.io/) for providing the foundational code and concepts used in the `splatfacto.py` implementation.


