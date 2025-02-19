import streamlit as st
import pickle

# List of available models
model_files = {
    "Linear Regression": "lr_reg.pkl",
    "Decision Tree": "d_tree.pkl",
    "K-Nearest Neighbors": "knn_b.pkl",
    "Naive Bayes": "nbayes.pkl",
    "Random Forest": "rf_cl.pkl",
    "Support Vector Machine": "svm_cr.pkl"
}

# Feature columns used for prediction
feature_columns = ['Age', 'Smokes', 'AreaQ', 'Alkhol']  # Replace with actual column names

# Function to load the selected model
def load_model(model_name):
    with open(model_files[model_name], 'rb') as file:
        model = pickle.load(file)
    return model

# Function to map numerical output to text labels
def map_prediction_to_label(prediction):
    if prediction == 0:
        return "No Lung Cancer Detected"
    elif prediction == 1:
        return "Lung Cancer Detected"
    else:
        return "Unknown Prediction"

# Streamlit input function for user input
def get_user_input(columns):
    user_input = {}
    
    st.write("Please enter the following details:")
    for col in columns:
        value = st.number_input(f"Enter value for {col}:", value=0.0)
        user_input[col] = value
    return user_input

# Home Page
def home_page():
    st.title("Lung Cancer Prediction App")
    st.write("""
    ### About Lung Cancer
    Lung cancer is one of the most common and serious types of cancer. It occurs when abnormal cells grow uncontrollably in the lungs. 
    Early detection and prediction can significantly improve treatment outcomes.

    #### Risk Factors:
    - Smoking
    - Exposure to secondhand smoke
    - Exposure to radon gas
    - Family history of lung cancer
    - Exposure to asbestos and other carcinogens

    #### Symptoms:
    - Persistent cough
    - Shortness of breath
    - Chest pain
    - Coughing up blood
    - Unexplained weight loss

    This app uses machine learning models to predict the likelihood of lung cancer based on input features like age, smoking habits, and more.
    """)

# Prediction Page
def prediction_page():
    st.title("Prediction Page")

    # Dropdown to select the model
    selected_model = st.selectbox("Select a model", list(model_files.keys()))

    # Load the selected model
    model = load_model(selected_model)

    # Get user input from the form
    user_input = get_user_input(feature_columns)

    # Button to make prediction
    if st.button("Make Prediction"):
        # Prepare the input for prediction (as a list of values)
        input_values = [[user_input[col] for col in feature_columns]]

        # Make the prediction
        prediction = model.predict(input_values)

        # Map the prediction to a text label
        prediction_label = map_prediction_to_label(prediction[0])

        # Display the prediction result
        st.write(f"Selected Model: {selected_model}")
        st.write(f"Prediction: {prediction_label}")

# Main function to render the app
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction"])

    # Display the selected page
    if page == "Home":
        home_page()
    elif page == "Prediction":
        prediction_page()

if __name__ == '__main__':
    main()