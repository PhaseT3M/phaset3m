# PhaseT3M

**PhaseT3M** is a reconstruction package designed for High-Resolution Transmission Electron Microscopy (HRTEM) data analysis.

### Features:
1. **Focal Series Reconstruction**  
   Enables precise reconstruction from focal series images.

2. **Three-Dimensional Phase Retrieval-based Tomography**  
   Combines multiple tilted focal series to perform 3D phase-contrast tomography.

---
This package follows the structure of the **[py4DSTEM](https://github.com/py4dstem/py4DSTEM)** package, ensuring compatibility and familiarity for users of similar tools.

## 🚀 Installation and Run

### 📦 Installation

Install the PhaseT3M

```bash
conda create -n PhaseT3M python = 3.11
conda activate PhaseT3M
conda install -c conda-forge cupy
pip install -e .
```

Run the example

```bash
python ./examples/test_focal_recon.ipynb
python ./examples/test_tilt_recon.ipynb
```
