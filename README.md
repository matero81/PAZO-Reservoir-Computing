# PAZO Reservoir Computing Code

This repository contains the code associated with the paper "[**Neuromorphic Light-Responsive Organic Matter for *in Materia* Reservoir Computing**](https://doi.org/10.1002/adma.202501813)," published in Advanced Materials (2025), DOI: 10.1002/adma.202501813.

This project provides the computational framework used to process experimental data and demonstrate the reservoir computing capabilities of the light-responsive organic material 'PAZO'. The code enables researchers to reproduce the key findings presented in the paper and apply the processing techniques to similar *in materia* computing experiments.

## Repository Contents

This repository contains the following main files:

* `functions_general.py`: A Python script containing general-purpose functions used for data processing and analysis within the project.
* `PAZO_network_processing.ipynb`: A Jupyter Notebook file that orchestrates the data loading, processing, analysis, and visualization steps described in the paper. This is the main script for running the experiments.
* `requirements.txt`: A file listing the necessary Python packages required to run the code.
* `MNIST_data/`: This directory contains the MNIST handwritten digits in a npz format.
* `Data to process/`: This directory contains the input data for the processing.
* `out_data/`: This directory contains the output plots and data after running the jupyter-notebook code

## Requirements

To run this code, you will need:

* **Python 3.11.0:** The code was developed and tested with this specific version. It is recommended to use Python 3.11.0 to ensure compatibility. You can download it from the [official Python website](https://www.python.org/downloads/release/python-3110/). Remember to add Python to your system's PATH during installation.
* **Jupyter Notebook environment:** An environment capable of running Jupyter Notebooks (`.ipynb` files).
* **Required Python packages:** The packages listed in `requirements.txt`.

## Recommended Setup

While not strictly mandatory, using an Integrated Development Environment (IDE) with good support for Python and Jupyter Notebooks is highly recommended for a smoother experience. [Visual Studio Code](https://code.visualstudio.com/download) is a popular choice with excellent [Jupyter Notebook support](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

It is also recommended to create a dedicated Python [virtual environment](https://code.visualstudio.com/docs/python/environments) for this project to manage dependencies and avoid conflicts with other Python projects.

## Quick Start Guide

Follow these steps to get the code up and running:

1.  **Clone the repository:** Clone this repository to your local machine using Git.
    ```bash
    git clone https://github.com/matero81/PAZO-Reservoir-Computing
    ```
2.  **Navigate to the project directory:** Open your terminal or command prompt and change your current directory to the cloned repository folder.
    ```bash
    cd PAZO-Reservoir-Computing
    ```
3.  **(Optional) Create and activate a virtual environment:**
    ```bash
    # Create environment
    python -m venv PAZO_python_environment
    # Activate environment (on Windows)
    .\PAZO_python_environment\Scripts\activate
    # Activate environment (on macOS/Linux)
    source PAZO_python_environment/bin/activate
    ```
4.  **Install dependencies:** Install the required Python packages using pip.
    ```bash
    pip install -r requirements.txt
    ```
    * **Note for Windows users:** If you encounter issues with long path names during package installation on Windows, you might need to enable long path support. Refer to this [guide](https://www.microfocus.com/documentation/filr/filr-4/filr-desktop/t47bx2ogpfz7.html) for instructions.
5.  **Open the project in your IDE:** Launch your preferred IDE (e.g., Visual Studio Code) and open the project folder.
6.  **Open and run the Jupyter Notebook:** Open the `PAZO_network_processing.ipynb` file in your IDE. Ensure that your IDE is using the correct Python environment (especially if you created a virtual environment).
7.  **Execute cells:** Run the code cells within the Jupyter Notebook sequentially to perform the data processing and analysis steps. The repository includes sample data in the `Data to process/` directory, allowing you to run the notebook out-of-the-box.
