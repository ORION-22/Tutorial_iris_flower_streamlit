import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

st.title("Machine learning - Iris")

sepal_length = st.slider('Sepal_length', 0.1, 7.9, 2.0)
sepal_width = st.slider('Sepal_width', 0.1, 7.9, 2.0)
petal_length = st.slider('petal_length', 0.1, 7.9, 2.0)
petal_width = st.slider('petal_width', 0.1, 7.9, 2.0)

iris = load_iris()
x = iris.data
y = iris.target

model = DecisionTreeClassifier()
model.fit(x, y)  # Training

y = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
y = iris.target_names[y[0]]
st.text(f"the flower is {y} ")
