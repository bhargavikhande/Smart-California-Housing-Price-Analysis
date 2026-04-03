import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Housing Intelligence", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")

    # Fill missing values
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

    # Feature Engineering
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    return df

df = load_data()

st.title("🏠 AI Real Estate Intelligence System")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Enter Property Details")

income = st.sidebar.slider("Median Income", 0.5, 15.0, 3.0)
rooms = st.sidebar.slider("Total Rooms", 100, 10000, 2000)
bedrooms = st.sidebar.slider("Total Bedrooms", 50, 5000, 500)
population = st.sidebar.slider("Population", 100, 10000, 1000)
households = st.sidebar.slider("Households", 50, 5000, 500)
age = st.sidebar.slider("House Age", 1, 50, 20)

ocean = st.sidebar.selectbox(
    "Ocean Proximity",
    df['ocean_proximity'].unique()
)

# =========================
# PREPARE DATA
# =========================
df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])

X = df_encoded.drop("median_house_value", axis=1)
y = df_encoded["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# =========================
# CREATE INPUT DATA
# =========================
input_dict = {
    'longitude': -120.0,
    'latitude': 35.0,
    'housing_median_age': age,
    'total_rooms': rooms,
    'total_bedrooms': bedrooms,
    'population': population,
    'households': households,
    'median_income': income,
    'rooms_per_household': rooms/households if households else 0,
    'bedrooms_per_room': bedrooms/rooms if rooms else 0,
    'population_per_household': population/households if households else 0
}

# Add ocean encoding
for col in X.columns:
    if "ocean_proximity" in col:
        input_dict[col] = 1 if col.endswith(ocean) else 0

input_df = pd.DataFrame([input_dict])[X.columns]

# =========================
# PREDICTION
# =========================
prediction = model.predict(input_df)[0]

st.subheader("💰 Predicted House Price")
st.success(f"${int(prediction)}")

st.write(f"📊 Model Accuracy: {round(accuracy*100,2)}%")

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📊 Feature Importance")

importances = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(feat_df.set_index("Feature"))

# =========================
# VISUALS
# =========================
st.subheader("📈 Income vs Price")

fig, ax = plt.subplots()
ax.scatter(df['median_income'], df['median_house_value'])
ax.set_xlabel("Income")
ax.set_ylabel("Price")
st.pyplot(fig)

# =========================
# AI INSIGHTS
# =========================
st.subheader("🤖 AI Insights")

avg_income = df['median_income'].mean()
avg_price = df['median_house_value'].mean()

if income > avg_income:
    insight = "High income area → Higher price potential 📈"
else:
    insight = "Lower income → Budget-friendly region 💰"

if prediction > avg_price:
    market = "Premium Market Zone 🏆"
else:
    market = "Affordable Investment Zone 🏡"

st.write(f"""
### Key Insights:
- {insight}
- {market}
- Coastal areas generally have higher value
- Engineered features improved model accuracy

### Investment Advice:
Look for:
✔ Low income + rising price → Gentrification  
✔ High income + low price → Undervalued  

### Risk Level:
""")

if income < avg_income and prediction < avg_price:
    st.error("🔴 High Risk Area")
elif income > avg_income and prediction > avg_price:
    st.success("🟢 Low Risk Investment")
else:
    st.warning("🟡 Medium Risk")

# =========================
# GEOGRAPHIC MAP
# =========================
st.subheader("🗺 Geographic Price Distribution")

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(
    df['longitude'], df['latitude'],
    c=df['median_house_value'],
    s=df['population']/100
)

plt.colorbar(scatter)
st.pyplot(fig2)
