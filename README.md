# Estimation of State of Health (SoH) for Li-ion Batteries Using OCV Tests

## Description

## Description

This project aims to estimate the State of Health (SoH) of lithium-ion batteries using both Incremental Current Open Circuit Voltage (OCV) tests and Dynamic Driving Cycle tests (Dynamic Stress Test, Federal Urban Driving Schedule, US06). 
The collected data is processed and used to train machine learning regression models to predict the remaining capacity and performance degradation. 
The results help evaluate battery reliability and support further optimization of battery management systems.

## Technologies Used

- Python 3.13.2
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Data Structure

Raw test data is stored in `.xlsx` files located under:
../data/raw/OCV-SOC/Incremental_Current_OCV/
../data/raw/Dynamic_tests/

The OCV dataset includes:

- Sample 1 at 0°C, 25°C, 45°C
- Sample 2 at 0°C, 25°C, 45°C

The Dynamic dataset includes:

- Dynamic Stress Test
- Federal Urban Driving Schedule
- US06

Data is loaded and preprocessed using a custom load_data_to_dict function.

Data cache is also included and stored in `loaded_files_cache.pkl` located under:
../data/cache/

## How to Run:

1. Clone the repository:

git clone <repository_url>
cd <repository_folder>

2. Install dependencies:

pip install -r requirements.txt

3. Launch Jupyter Notebook:

jupyter notebook

4. Open `notebook.ipynb` and run the cells step by step.

## Expected Results

The notebook computes the battery SoH and evaluates the estimation accuracy using standard metrics (e.g., Mean Absolute Error). 
Visualizations are provided for better interpretation.

## License

This project is developed for academic purposes. License terms can be specified here if applicable.

## Author

Nikolai Matgafurov
Institution: Saint Petersburg Electrotechnical University
Contact: valak3j3@gmail.com