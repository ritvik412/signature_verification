# Signature Verification with Sigma-Lognormal Modeling

![plot](https://github.com/user-attachments/assets/34d358d5-6df6-46ff-8bd6-e65f009ae52b)

## Project Overview

This project focuses on analyzing handwriting signatures using sigma-lognormal modeling to extract handstroke characteristics. The implementation involves preprocessing signature data, applying sigma-lognormal transformations, and visualizing results to enable signature verification and forgery detection. The system processes signature coordinate data to model the velocity profiles of handwriting strokes, which can be used for biometric authentication.

---

## 🗂️ Dataset

Place your raw signature text files under the following structure:

```bash
Data/
└── 20U/
    ├── u0001_g_0101v00.txt
    ├── u0002_g_0101v00.txt
    └── ...
```

Each file should follow the format:

->Line 1: number of points (N)

->Lines 2–N+1: `x y t [pressure tilt,...]`

## 🔧 *Installation & Setup*

# Clone the repo
git clone https://github.com/ritvik412/signature-verification.git
cd signature-verification

# Create and activate a virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install required Python packages
pip install -r requirements.txt

# Install local sigma-lognormal library
pip install -e ./sigma_lognormal

## 📖 *Usage*
Launch the Jupyter notebook: demo.ipynb
