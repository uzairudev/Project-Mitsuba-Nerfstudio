# Spectral Data Generation using Mitsuba and TOUCAN Multispectral Camera

This repository contains code and data for generating spectral data using the Mitsuba renderer and the TOUCAN Multispectral Camera. The project is organized into several Jupyter notebooks that handle different spectral bands, as well as encoded bands.

## Repository Structure

```
.
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

## Project Overview

This project aims to generate spectral data using a custom TOUCAN Multispectral Camera model in the Mitsuba renderer. The generated data spans across nine individual spectral bands, as well as three sets of encoded bands.

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

- **lego/**: Contains the `cbox.xml` scene file, along with all necessary Lego meshes and textures for rendering.
- **my_first_render.png**: Example render image.
- **pexels-fwstudio-33348-172289.jpg**: Reference image used in the project.

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

## Usage

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. **Install required dependencies**:
   Ensure you have Jupyter Notebook and the necessary Python libraries installed (e.g., `matplotlib`, `numpy`).

3. **Navigate through the Jupyter notebooks**:
   Open and run the Jupyter notebooks to generate the spectral data for different bands.

4. **Render the scenes**:
   Use Mitsuba to render the scenes specified in the `lego/cbox.xml` file using the spectral sensitivity settings.


## Acknowledgements

- [Mitsuba Renderer](https://www.mitsuba-renderer.org/) for providing the rendering framework.
- The authors of any referenced images or code snippets used in this project.
