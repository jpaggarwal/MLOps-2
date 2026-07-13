import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download
import joblib

# Download model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="jpaggarwal/churn-model",
    filename="best_churn_model.joblib"
)

# Load model
model = joblib.load(model_path)

classification_threshold = 0.45


def predict_churn(
    CreditScore,
    Geography,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    HasCrCard,
    IsActiveMember,
    EstimatedSalary,
):

    input_data = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": 1 if HasCrCard == "Yes" else 0,
        "IsActiveMember": 1 if IsActiveMember == "Yes" else 0,
        "EstimatedSalary": EstimatedSalary,
    }])

    prediction_proba = model.predict_proba(input_data)[0, 1]

    prediction = int(prediction_proba >= classification_threshold)

    if prediction == 1:
        return (
            "⚠️ Customer is likely to CHURN",
            f"{prediction_proba:.2%}"
        )
    else:
        return (
            "✅ Customer is NOT likely to churn",
            f"{prediction_proba:.2%}"
        )


demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Credit Score", value=650),
        gr.Dropdown(
            ["France", "Germany", "Spain"],
            label="Geography",
            value="France"
        ),
        gr.Number(label="Age", value=30),
        gr.Number(label="Tenure", value=5),
        gr.Number(label="Account Balance", value=10000),
        gr.Number(label="Number of Products", value=1),
        gr.Radio(
            ["Yes", "No"],
            label="Has Credit Card?",
            value="Yes"
        ),
        gr.Radio(
            ["Yes", "No"],
            label="Is Active Member?",
            value="Yes"
        ),
        gr.Number(label="Estimated Salary", value=50000),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability of Churn"),
    ],
    title="🏦 Bank Customer Churn Prediction",
    description=(
        "Predict whether a customer is likely to churn based on "
        "their demographic and banking information."
    ),
    flagging_mode="never"
)

demo.launch()
