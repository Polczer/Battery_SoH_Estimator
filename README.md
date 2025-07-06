# Estimation of State of Health (SoH) for Li-ion Batteries Using OCV Tests

## Description
This project estimates the State of Health (SoH) of lithium-ion batteries using:
- Incremental Current Open Circuit Voltage (OCV) tests  
- Dynamic Driving Cycle tests:
  - Dynamic Stress Test (DST)
  - Federal Urban Driving Schedule (FUDS)
  - US06

Machine learning models predict remaining capacity and degradation patterns to optimize Battery Management Systems (BMS).

## Technologies
- **Python 3.13.2** (with `requirements.txt` for dependencies)
- **Core Libraries**:
  ```python
  pandas, numpy, scikit-learn, matplotlib, seaborn
  ```
- **Environment**: Jupyter Notebook

## Data Structure
```
Battery_SoH_Estimator/
├── data/
│   ├── raw/
│   │   ├── OCV-SOC/Incremental_Current_OCV/  # XLSX files for:
│   │   │   ├── Sample1 (0°C, 25°C, 45°C)
│   │   │   └── Sample2 (0°C, 25°C, 45°C)
│   │   └── Dynamic_tests/  # DST, FUDS, US06
│   └── cache/loaded_files_cache.pkl  # Preprocessed data
└── notebook.ipynb  # Main analysis
```

## Installation & Usage
1. **Clone and setup**:
   ```bash
   git clone https://github.com/Polczer/Battery_SoH_Estimator.git
   cd Battery_SoH_Estimator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch analysis**:
   ```bash
   jupyter notebook
   ```
   - Open `notebook.ipynb`
   - Run cells sequentially

## Expected Outputs
- SoH estimation metrics (MAE, RMSE)
- Degradation visualizations:
  - Capacity fade curves
  - Voltage hysteresis plots
  - Feature importance diagrams

## License
Academic use only. For commercial applications, please contact the author.

## Contact
**Nikolai Matgafurov**  
[Saint Petersburg Electrotechnical University]  
📧 valak3j3@gmail.com