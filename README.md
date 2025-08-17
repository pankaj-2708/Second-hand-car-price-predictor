# 🚗 Second-hand Car Price Predictor

An interactive machine learning web app that **predicts the resale value
of a used car** based on **16 different inputs** provided by the user.\
The app is built for easy use by anyone --- just enter details like
brand, model, year, mileage, fuel type, etc., and instantly get the
predicted market value.

------------------------------------------------------------------------

## ✨ Features

-   📊 Predicts the **resale price of a second-hand car** with ML
    models\
-   🖥️ User-friendly **Streamlit interface**\
-   🔄 End-to-end **ML pipeline** (data preprocessing → training →
    evaluation → prediction)\
-   📦 Experiment tracking with **MLflow**\
-   📂 Versioning of datasets & models using **DVC (Data Version
    Control)**\
-   🛠️ Structured using **Cookiecutter Data Science** template\
-   ⚡ Built on top of **scikit-learn** with pipelines and preprocessing

------------------------------------------------------------------------

## 🚀 Live Demo

[🔗 **Click here to try the app**](https://second-hand-car-price-predictor-qhpy2ulq94r8bvwkwcciwr.streamlit.app)

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   **Frontend/UI:** Streamlit\
-   **Modeling & ML:** scikit-learn\
-   **Experiment Tracking:** MLflow\
-   **Data & Model Versioning:** DVC\
-   **Project Structure:** Cookiecutter Data Science\
-   **Language:** Python 3.10+

------------------------------------------------------------------------

## ⚙️ Installation & Setup

Clone the repository:

``` bash
git clone https://github.com/pankaj-2708/Second-hand-car-price-predictor.git
cd Second-hand-car-price-predictor
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the Streamlit app:

``` bash
streamlit run ./second_hand_car_price_prediction/Frontend/main.py
```

------------------------------------------------------------------------

## 📈 Workflow

1.  **Data Collection** -- Gather car price dataset\
2.  **Feature Engineering** -- Process 16 input features\
3.  **Model Training** -- Train ML models with scikit-learn\
4.  **Experiment Tracking** -- Log experiments in MLflow\
5.  **Versioning** -- Track datasets and models with DVC\
6.  **Deployment** -- Expose via Streamlit app

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the app: 1. Fork the
repo\
2. Create a new branch (`feature-new`)\
3. Commit your changes\
4. Open a Pull Request

------------------------------------------------------------------------

## 📬 Contact

Created with ❤️ by **[pankaj-2708](https://github.com/pankaj-2708)**\
Feel free to open issues or suggestions in the repository.

------------------------------------------------------------------------

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data                (not shown here beacuse it is getting tracked by dvc)
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries (tracked by dvc)
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         second_hand_car_price_prediction and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── second_hand_car_price_prediction   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes second_hand_car_price_prediction a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

