  ---
  title: Battery RUL API
  sdk: docker
  app_port: 8000
  ---

**Battery Health and Remaining Useful Life Prediction Using Lightweight Machine Learning Models**
https://doi.org/10.1109/ICIMCIS68501.2025.11327063

**Overview**
Battery degradation has become a crucial challenge to environmental safety and energy management systems due to the limited lifespan of lithium-ion batteries and their improper handling. This project aims to develop lightweight machine learning models to estimate the remaining useful life (RUL) of lithium-ion batteries, aligning with SDG 7 (Affordable and Clean Energy) and SDG 12 (Responsible Consumption and Production) to support responsible battery usage. We implement and compare five regression models to predict battery RUL, such as Gradient Boosting, XGBoost, Random Forest, Decision Tree, and Polynomial Regression.

**Live Interactive Deployment**
Interactive Streamlit web application to estimate battery RUL using Gradient Boosting. Web app link: https://battery-rul.streamlit.app/

**Project Structure**
```
batteryRUL/
└── app/                     # FastAPI application
    ├── __init__.py
    ├── main.py
    └── schemas.py
│
├── battery_rul_model.pkl    # Pickle of gradient boosting
├── Battery_RUL.csv          # Battery RUL dataset
├── docker-compose.yml       # Container configuration
├── dockerfile               # Docker build definition
├── Model.ipynb              # Model notebook
├── README.md                # Project documentation
├── requirements.txt         # Required dependencies
│
└── results/                 # Evaluation plots and visualizations
    ├── feature_importances_decision_tree.png
    ├── feature_importances_gradient_boosting.png
    ├── feature_importances_random_forest.png
    ├── feature_importances_xgboost.png
    ├── heatmap.png
    ├── scatterplot_decision_tree.png
    ├── scatterplot_gradient_boosting.png
    ├── scatterplot_polynomial.png
    ├── scatterplot_random_forest.png
    └── scatterplot_xgboost.png
```

**Acknowledgments**
- Dataset source: https://www.kaggle.com/datasets/ignaciovinuales/battery-remaining-useful-life-rul
- This project supports Sustainable Development Goals 7 & 12

**Contact**
Dionisius Sylvester Wime - dionisius.wime@binus.ac.id
