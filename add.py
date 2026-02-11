import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="KNN Weather Classifier")
st.title("KNN Weather Classification")

# Dataset (Temperature, Humidity)
X = np.array([
    [50, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [20, 75]
])

# Labels (0 = Sunny, 1 = Rainy)
y = np.array([0, 1, 0, 0, 1, 1])
label_map = {0: "Sunny", 1: "Rainy"}

# Sidebar Inputs
st.sidebar.header("Input Features")
temp = st.sidebar.slider("Temperature", 10, 60, 26)
hum = st.sidebar.slider("Humidity", 50, 95, 78)
k_value = st.sidebar.slider("K (Number of Neighbors)", 1, 5, 3)

# Scaling (IMPORTANT for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_scaled, y)

# New Data Prediction
new_point = np.array([[temp, hum]])
new_point_scaled = scaler.transform(new_point)
prediction = knn.predict(new_point_scaled)[0]

st.subheader(f"Predicted Weather: {label_map[prediction]}")

# Accuracy (on training data)
accuracy = knn.score(X_scaled, y)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Get Nearest Neighbors
distances, indices = knn.kneighbors(new_point_scaled)
neighbors = X[indices[0]]

# Plot
fig, ax = plt.subplots()

# Plot Sunny
ax.scatter(
    X[y == 0, 0], X[y == 0, 1],
    color='orange', label='Sunny', s=100, edgecolor='black'
)

# Plot Rainy
ax.scatter(
    X[y == 1, 0], X[y == 1, 1],
    color='blue', label='Rainy', s=100, edgecolor='black'
)

# Plot Neighbors
ax.scatter(
    neighbors[:, 0], neighbors[:, 1],
    facecolors='none', edgecolors='green',
    s=300, linewidth=2, label='Nearest Neighbors'
)

# Plot New Point
ax.scatter(
    temp, hum,
    color='red' if prediction == 1 else 'orange',
    marker='*', s=300,
    edgecolor='black',
    label=f'New Day ({label_map[prediction]})'
)

ax.set_xlabel("Temperature")
ax.set_ylabel("Humidity")
ax.set_title("KNN Weather Classification")
ax.legend()
ax.grid(True)

st.pyplot(fig)
