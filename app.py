import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Real Estate Investment Dashboard", layout="wide")

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv('real_estate_price_size_year.csv')

current_year = datetime.datetime.now().year
df['property_age'] = current_year - df['year']
df['price_per_size'] = df['price'] / df['size']

# Fake property detection
df['price_zscore'] = np.abs(stats.zscore(df['price_per_size']))
df['is_fake'] = df.apply(
    lambda row: 1 if row['price_zscore']>3 or row['property_age']<0 or row['property_age']>200 else 0,
    axis=1
)

# Random external features (simulating MLS, tax, zoning)
np.random.seed(42)
df['tax_amount'] = np.random.randint(1000, 15000, len(df))
df['zoning_type'] = np.random.choice(['residential','commercial','industrial'], len(df))
df['is_off_market'] = np.random.choice([0,1], len(df))
df['num_nearby_listings'] = np.random.randint(0,20, len(df))

# Simulate estimated rent
df['estimated_rent'] = np.random.randint(500, 5000, len(df))

# Investment metrics
df['NOI'] = df['estimated_rent'] * 12 * 0.8
df['cap_rate'] = df['NOI'] / df['price']
df['cash_invested'] = df['price'] * 0.2
df['cash_on_cash_return'] = df['NOI'] / df['cash_invested']
df['ROI'] = (df['estimated_rent'] * 12) / df['price']
df['predicted_rent_growth'] = np.random.uniform(0.01, 0.1, len(df))
df['future_yield'] = df['ROI'] * (1 + df['predicted_rent_growth'])
# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.title("Filters")
zoning_filter = st.sidebar.multiselect("Zoning Type", options=df['zoning_type'].unique(), default=df['zoning_type'].unique())
offmarket_filter = st.sidebar.selectbox("Off-Market Status", options=["All", "On Market", "Off Market"])
age_filter = st.sidebar.slider("Property Age (years)", int(df['property_age'].min()), int(df['property_age'].max()), (int(df['property_age'].min()), int(df['property_age'].max())))

filtered_df = df[df['zoning_type'].isin(zoning_filter)]
if offmarket_filter == "On Market":
    filtered_df = filtered_df[filtered_df['is_off_market']==0]
elif offmarket_filter == "Off Market":
    filtered_df = filtered_df[filtered_df['is_off_market']==1]

filtered_df = filtered_df[(filtered_df['property_age']>=age_filter[0]) & (filtered_df['property_age']<=age_filter[1])]

# -----------------------
# Display top properties
# -----------------------
st.title("Top Real Estate Investment Properties")
top_properties = filtered_df[filtered_df['is_fake']==0].sort_values('future_yield', ascending=False).head(10)
st.dataframe(top_properties[['price','size','year','estimated_rent','ROI','cap_rate','cash_on_cash_return','future_yield','zoning_type','is_off_market']])

# -----------------------
# Visualizations
# -----------------------
st.subheader("Price per Size Distribution")
fig1, ax1 = plt.subplots()
ax1.hist(filtered_df['price_per_size'], bins=30, color='blue', alpha=0.7)
ax1.set_xlabel('Price per Size')
ax1.set_ylabel('Count')
st.pyplot(fig1)

st.subheader("Fake vs Real Properties")
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(filtered_df['size'], filtered_df['price'], c=filtered_df['is_fake'], cmap='bwr', alpha=0.7)
ax2.set_xlabel('Size')
ax2.set_ylabel('Price')
st.pyplot(fig2)

st.subheader("Price per Size vs Property Age")
fig3, ax3 = plt.subplots()
ax3.scatter(filtered_df['property_age'], filtered_df['price_per_size'], c=filtered_df['is_fake'], cmap='bwr', alpha=0.7)
ax3.set_xlabel('Property Age')
ax3.set_ylabel('Price per Size')
st.pyplot(fig3)

st.subheader("Top 10 Investment Properties - Future Yield")
fig4, ax4 = plt.subplots()
ax4.bar(top_properties['size'], top_properties['future_yield'], color='green', alpha=0.7)
ax4.set_xlabel('Size')
ax4.set_ylabel('Future Yield')
st.pyplot(fig4)
