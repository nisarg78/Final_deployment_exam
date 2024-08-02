import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


st.title("Machine Learning Application with Iris Dataset")

option = st.radio(
    "Choose a dataset",
    ("Use Iris Dataset", "Upload Your Dataset")
)

if option == "Use Iris Dataset":
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    st.write("Using Iris Dataset")
    st.write(data.head())

elif option == "Upload Your Dataset":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset")
        st.write(data.head())
    else:
        st.write("Please upload a CSV file to proceed.")


if 'data' in locals():
    # Select features and target
    features = st.multiselect("Select features", data.columns.tolist(), default=data.columns.tolist()[:-1])
    target = st.selectbox("Select target", data.columns.tolist(), index=len(data.columns.tolist()) - 1)

    if features and target:
        X = data[features]
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Show results
        st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
        st.write("Confusion Matrix:", confusion_matrix(y_test, predictions))

        # Plot results
        fig, ax = plt.subplots()
        ax.bar(['Correct', 'Incorrect'], [accuracy_score(y_test, predictions), 1 - accuracy_score(y_test, predictions)])
        st.pyplot(fig)
else:
    st.write("No dataset loaded.")