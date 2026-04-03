# 🏠 Smart California Housing Price Analysis

## Overview
The **Smart California Housing Price Analysis** is an interactive AI-powered web application built with Streamlit. It leverages machine learning (Random Forest) and data analysis techniques to predict housing prices in California and provide actionable real-estate investment insights.

This project goes beyond simple predictions by offering feature engineering, comprehensive data visualizations, geographic price distribution analysis, and an automated risk-assessment engine based on user inputs. 

## ✨ Features
- **🏡 Price Prediction**: Accurately estimates median house values using a trained `RandomForestRegressor` model based on user-provided property and location details.
- **📊 Feature Importance**: Visualizes which factors (e.g., location, median income, ocean proximity) most strongly influence housing prices.
- **📈 Income vs Price Analysis**: Displays scatter plots showing the correlation between income and property values.
- **🤖 AI Insights & Investment Advice**: Automatically categorizes the property into market zones (e.g., "Premium Market Zone", "Affordable Investment Zone") and evaluates investment risk (Low/Medium/High).
- **🗺️ Geographic Visualization**: Scatter mapping to visualize property value distribution and population density across California coordinates.
- **🛠️ Advanced Preprocessing**: Includes median imputation for missing values and custom calculated metrics (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) to enhance model accuracy.

## 🛠️ Technologies Used
- **Python**: Core programming language.
- **Streamlit**: Web framework for building the interactive UI.
- **Pandas & NumPy**: Data manipulation, preprocessing, and feature engineering.
- **Scikit-Learn**: Machine learning (Random Forest regression, train-test splitting).
- **Matplotlib**: Data visualization and plotting.

## 🚀 Installation and Setup

### Prerequisites
Make sure you have Python installed on your system.

### 1. Clone the repository
```bash
# If you haven't cloned it yet
git clone <repository_url>
cd "Smart California Housing Price Analysis"
```

### 2. Install dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Run the application
Start the Streamlit development server:
```bash
streamlit run app.py
```
This will open the application in your default web browser (typically at `http://localhost:8501`).

## 📁 Project Structure

- `app.py`: The main Streamlit application containing the UI, data processing, machine learning pipeline, and visualization logic.
- `housing.csv`: The California housing dataset used to train the model and generate real-time insights.
- `Project_Report.md`: A detailed academic report containing the abstract, methodology, architectural diagram, and extensive documentation of the analytical approach.
- `requirements.txt`: List of required Python dependencies.

## 💡 Usage
1. Open the application in your web browser.
2. Use the **sidebar** to adjust property details: Median Income, Total Rooms, Total Bedrooms, Population, Households, House Age, and Ocean Proximity.
3. The main dashboard will instantly update to display:
   - The predicted house price along with model accuracy.
   - Key AI Insights and an Investment Advice risk assessment.
   - Interactive visual charts (Feature Importance, Income vs Price).
   - Geographic price distribution map.

## 📝 Background
This project focuses on identifying key socioeconomic and geographical factors (like income, population density, and distance to the coast) that affect property values, utilizing these insights to empower smart real estate investment decisions.
