# ⚡ Electric Propulsion System Digital Twin

[![MATLAB](https://img.shields.io/badge/MATLAB-R2025b-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Simscape](https://img.shields.io/badge/Simscape-Electrical-orange.svg)](https://www.mathworks.com/products/simscape-electrical.html)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehasardesai/Electric-Propulsions-System-Digital-Twin/blob/main/Electric_Propulsion_Fault_Detection_Workshop.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete digital twin implementation for electric propulsion systems (e.g., electric aircraft, UAVs) with **machine learning-based fault detection**. This project demonstrates the full workflow from physical system modeling to ML-powered predictive maintenance.

![System Overview](https://img.shields.io/badge/Battery-201.6V-blue) ![Motor](https://img.shields.io/badge/Motor-20kW-orange) ![Propeller](https://img.shields.io/badge/Propeller-0.9m-green) ![Thrust](https://img.shields.io/badge/Thrust-139N-red)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Fault Detection](#-fault-detection)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Workshop Materials](#-workshop-materials)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project creates a **digital twin** of an electric propulsion system using MATLAB Simscape, then uses the virtual model to:

1. **Simulate normal operation** and collect healthy system data
2. **Inject faults** (battery degradation, motor efficiency loss, propeller damage)
3. **Generate labeled datasets** for machine learning
4. **Train ML models** to detect and classify faults in real-time

### Why Digital Twins for Fault Detection?

| Traditional Approach | Digital Twin Approach |
|---------------------|----------------------|
| Test on real hardware | Safe virtual testing |
| Expensive fault injection | Zero-cost fault simulation |
| Limited fault scenarios | Unlimited fault variations |
| Risk of equipment damage | No physical risk |

---

## 🏗️ System Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Battery   │────▶│  Motor & Drive  │────▶│  Propeller  │────▶ Thrust
│   201.6V    │     │  20kW / 88% eff │     │   0.9m dia  │      ~139N
│   100Ah     │     │   2500 RPM      │     │  Kt = 0.10  │
└─────────────┘     └─────────────────┘     └─────────────┘
       │                    │                      │
       ▼                    ▼                      ▼
   ┌───────┐           ┌────────┐            ┌─────────┐
   │Voltage│           │ Speed  │            │ Thrust  │
   │Current│           │ Torque │            │         │
   └───────┘           └────────┘            └─────────┘
       │                    │                      │
       └────────────────────┴──────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  ML Fault       │
                   │  Predictor      │
                   │  (Random Forest)│
                   └─────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Fault Classification:    │
              │ • Healthy                │
              │ • Battery Fault          │
              │ • Motor Fault            │
              │ • Propeller Fault        │
              └──────────────────────────┘
```

---

## ✨ Features

### Simscape Model
- ⚡ Complete electric propulsion system model
- 🔋 Battery with configurable internal resistance
- 🔧 Motor & Drive with efficiency modeling
- 🌀 Propeller aerodynamics (thrust & torque)
- 📊 Comprehensive sensor instrumentation

### Fault Injection
- 🔴 **Battery Fault**: Increased internal resistance (0.05Ω → 0.15Ω)
- 🟠 **Motor Fault**: Efficiency degradation (88% → 70%)
- 🔵 **Propeller Fault**: Blade damage / reduced thrust coefficient (Kt: 0.10 → 0.06)

### Machine Learning
- 🤖 6 ML models compared (Random Forest, SVM, Neural Network, etc.)
- 📈 Feature engineering for improved accuracy
- 📉 Confusion matrix and classification reports
- 🎛️ Interactive prediction demo with sliders

---

## 🛠️ Installation

### Prerequisites

#### MATLAB (R2025b)
```
Required Toolboxes:
├── Simulink
├── Simscape
├── Simscape Electrical
└── Statistics and Machine Learning Toolbox (optional)
```

Verify installation:
```matlab
>> ver
```

#### Python (for ML - or use Google Colab)
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### Clone Repository
```bash
git clone https://github.com/nehasardesai/Electric-Propulsions-System-Digital-Twin.git
cd Electric-Propulsions-System-Digital-Twin
```

---

## 🚀 Usage

### Step 1: Run the Simscape Model

```matlab
% Open MATLAB and navigate to project folder
cd Electric-Propulsions-System-Digital-Twin

% Set workspace variables
Kt_prop = 0.10;      % Propeller thrust coefficient
throttle_cmd = 76;   % Torque command (N·m)

% Open and run the model
open_system('ElectricPropSys_digitalTwin.slx')
sim('ElectricPropSys_digitalTwin')
```

**Expected Output:** ~2500 RPM, ~139 N thrust at steady state

### Step 2: Generate Fault Dataset

```matlab
% Run the data generation script
run('simscape_fault_dataset_generator.m')

% This will:
% - Inject 4 fault conditions (healthy + 3 faults)
% - Run 50 simulations per condition
% - Save data to simscape_fault_dataset.csv
```

### Step 3: Train ML Model

**Option A: Google Colab (Recommended)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehasardesai/Electric-Propulsions-System-Digital-Twin/blob/main/Electric_Propulsion_Fault_Detection_Workshop.ipynb)

**Option B: Local Python**
```bash
python fault_predictor.py
```

### Step 4: Real-Time Prediction

```python
from fault_predictor import predict_fault
import joblib

# Load trained model
model = joblib.load('fault_predictor_model.joblib')
scaler = joblib.load('fault_predictor_scaler.joblib')

# New sensor measurements
measurements = {
    'Throttle': 0.75,
    'Voltage': 185.0,      # Low voltage indicates battery fault
    'Current': 110.0,
    'Speed_RPM': 2300,
    'Torque': -65.0,
    'Thrust': 120.0,
    'Power_Elec': 20350,
    'Power_Mech': 15650,
    'Efficiency': 85.0
}

# Predict
result = predict_fault(measurements, model, scaler)
print(f"Detected Fault: {result['fault_name']}")
print(f"Confidence: {result['confidence']:.1%}")
```

---

## 🔍 Fault Detection

### Fault Signatures

| Fault | Label | Key Indicator | Observable Change |
|-------|-------|---------------|-------------------|
| **Healthy** | 0 | All nominal | Normal operation |
| **Battery** | 1 | Voltage drop | Lower V at same current |
| **Motor** | 2 | Low efficiency | High P_elec, low P_mech |
| **Propeller** | 3 | Low thrust | Reduced thrust at same RPM |

### Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | 95-100% | 0.95+ |
| SVM (RBF) | 95-100% | 0.95+ |
| Neural Network | 93-100% | 0.93+ |
| Gradient Boosting | 90-95% | 0.90+ |
| Decision Tree | 85-90% | 0.85+ |
| KNN | 75-85% | 0.75+ |

### Top Features for Classification

1. **Thrust_per_Power** - Propeller health indicator
2. **Power_Ratio** - Overall system efficiency
3. **Voltage_Drop_Pct** - Battery health indicator
4. **Efficiency** - Motor health indicator
5. **Mean Voltage** - Direct battery measurement

---

## 📁 Project Structure

```
Electric-Propulsions-System-Digital-Twin/
│
├── 📂 MATLAB/
│   ├── ElectricPropSys_digitalTwin.slx    # Simscape model
│   ├── simscape_fault_dataset_generator.m  # Data generation script
│   ├── model_setup_for_dataset.m          # Setup helper script
│   └── fault_detection_train_model.m      # MATLAB ML training
│
├── 📂 Python/
│   ├── fault_predictor.py                 # ML training script
│   ├── fault_predictor_model.joblib       # Trained model
│   ├── fault_predictor_scaler.joblib      # Feature scaler
│   └── fault_predictor_config.json        # Configuration
│
├── 📂 Data/
│   └── simscape_fault_dataset.csv         # Training dataset
│
├── 📂 Notebooks/
│   └── Electric_Propulsion_Fault_Detection_Workshop.ipynb  # Colab notebook
│
├── 📂 Docs/
│   ├── Workshop_Prework_Instructions.docx # Setup guide
│   └── Workshop_Instruction_Manual.docx   # Step-by-step manual
│
├── 📂 Images/
│   ├── fault_prediction_analysis.png      # Results visualization
│   └── model_comparison.png               # Model comparison chart
│
├── README.md                              # This file
└── LICENSE                                # MIT License
```

---

## 📊 Results

### Confusion Matrix (Random Forest)
```
                 Predicted
              H    B    M    P
Actual  H  [ 10    0    0    0 ]
        B  [  0   10    0    0 ]
        M  [  0    0   10    0 ]
        P  [  0    0    0   10 ]

H = Healthy, B = Battery, M = Motor, P = Propeller
```

### Feature Importance
![Feature Importance](Images/feature_importance.png)

---

## 📚 Workshop Materials

This repository includes complete workshop materials for teaching digital twin and ML concepts:

| Document | Description |
|----------|-------------|
| [Prework Instructions](https://github.com/nehasardesai/Electric-Propulsions-System-Digital-Twin/blob/main/Workshop_Prework_Instructions.docx) | Software setup checklist |
| [Instruction Manual](https://github.com/nehasardesai/Electric-Propulsions-System-Digital-Twin/blob/main/Workshop_Prework_Instructions.docx) | Step-by-step workshop guide |

### Workshop Agenda
1. **Part 1**: Build Simscape model (90 min)
2. **Part 2**: Generate fault data (30 min)
3. **Part 3**: Train ML classifier (45 min)
4. **Demo**: Interactive fault prediction (15 min)

---

## 🔬 Technical Details

### System Specifications

| Component | Parameter | Value |
|-----------|-----------|-------|
| Battery | Nominal Voltage | 201.6 V |
| Battery | Capacity | 100 Ah |
| Battery | Internal Resistance | 0.05 Ω |
| Motor | Max Power | 20 kW |
| Motor | Max Torque | 100 N·m |
| Motor | Efficiency | 88% |
| Motor | Rated Speed | 2500 RPM |
| Propeller | Diameter | 0.9 m |
| Propeller | Thrust Coefficient (Kt) | 0.10 |
| Propeller | Torque Coefficient (Kq) | 0.060 |

### Propeller Equations

```
Thrust: T = Kt × ρ × n² × D⁴
Torque: Q = Kq × ρ × n² × D⁵

Where:
  n = rotational speed (rev/s)
  ρ = air density (1.225 kg/m³)
  D = propeller diameter (m)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution
- [ ] Add more fault types (sensor faults, controller faults)
- [ ] Implement fault severity levels
- [ ] Add LSTM for time-series fault prediction
- [ ] Create real-time dashboard
- [ ] Hardware-in-the-loop integration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Neha Sardesai** - [@nehasardesai](https://github.com/nehasardesai)

Project Link: [https://github.com/nehasardesai/Electric-Propulsions-System-Digital-Twin](https://github.com/nehasardesai/Electric-Propulsions-System-Digital-Twin)

---

<p align="center">
  <b>⭐ Star this repository if you found it helpful! ⭐</b>
</p>
