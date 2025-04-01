import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Hello!", layout="wide")

# Sample data (embedded in the code since you provided it)
data = {
    "YearsExperience": [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.3, 3.8, 4.0,
                        4.1, 4.5, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9, 8.2, 8.7,
                        9.0, 9.5, 10.3],
    "Salary": [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445,
               57189, 63218, 55794, 56957, 66029, 83088, 81363, 93940, 91738,
               98273, 101302, 113812, 109431, 105582, 116969, 122391]
}

st.title("💰 Salary Prediction Tool")

# Option to upload or use default data
use_default = st.checkbox("Use sample data (provided)", value=True)

if use_default:
    df = pd.DataFrame(data)
else:
    uploaded_file = st.file_uploader("Or upload your CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

st.write("### Data Preview", df.head())

# Extract features
x = df[["YearsExperience"]]
y = df['Salary']

# Polynomial degree selector
degree = st.slider("Select polynomial degree", 1, 10, 3,
                   help="Higher degrees may overfit the data!")

# Transform data and fit model
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)

# Generate smooth curve for visualization
x_range = np.linspace(x.min().values[0], x.max().values[0], 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_pred = model.predict(x_range_poly)

# User prediction
st.write("---")
col1, col2 = st.columns(2)
with col1:
    customer_exp = st.number_input(
        "Enter years of experience",
        min_value=float(x.min()),
        max_value=float(x.max()),
        value=float(x.mean())
    )
with col2:
    if st.button("Predict Salary"):
        custom_pred = model.predict(poly.transform([[customer_exp]]))[0]
        st.success(f"Predicted Salary: **₹{custom_pred:,.2f}**")
    st.write("After entering the value click the button....!!")
# Create interactive plot
fig = go.Figure()

# Actual data points
fig.add_trace(
    go.Scatter(
        x=x["YearsExperience"],
        y=y,
        mode='markers+text',
        name='Actual Salaries',
        marker=dict(color='#636EFA', size=10, line=dict(width=1, color='DarkSlateGrey')),
        text=[f"₹{val:,.0f}" for val in y],
        textposition="top center",
        hoverinfo="x+y+text"
    )
)

# Regression line
fig.add_trace(
    go.Scatter(
        x=x_range.flatten(),
        y=y_pred,
        mode='lines',
        name=f'Degree {degree} Fit',
        line=dict(color='#FF7F0E', width=3),
        hoverinfo="none"
    )
)

# Highlight user's prediction if available
if 'custom_pred' in locals():
    fig.add_trace(
        go.Scatter(
            x=[customer_exp],
            y=[custom_pred],
            mode='markers+text',
            name='Your Prediction',
            marker=dict(color='#2CA02C', size=15, symbol='diamond'),
            text=[f"Predicted: ₹{custom_pred:,.0f}"],
            textposition="bottom right"
        )
    )

# Layout customization
fig.update_layout(
    title='<b>Salary vs Experience</b>',
    xaxis_title='Years of Experience',
    yaxis_title='Annual Salary',
    hoverlabel=dict(bgcolor="white", font_size=12),
    template='plotly_white',
    height=600
)

# Add reference lines
fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
fig.update_yaxes(
    showgrid=True,
    gridwidth=0.5,
    gridcolor='LightGrey',
    tickprefix="₹",
    tickformat=","
)

st.plotly_chart(fig, use_container_width=True)

# Model metrics
st.write("---")
st.write("### Model Details")
st.code(f"R² Score: {model.score(x_poly, y):.3f}")
