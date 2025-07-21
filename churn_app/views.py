from django.shortcuts import render
from .forms import UploadCSVForm
import pandas as pd
import pickle
import os
from django.http import HttpResponse
import csv
from collections import Counter

# Load model and features once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # churn_app/
PROJECT_DIR = os.path.join(BASE_DIR, '..')  # customer_churn_django/

MODEL_PATH = os.path.join(PROJECT_DIR, 'churn_model.pkl')
FEATURES_PATH = os.path.join(PROJECT_DIR, 'model_features.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    model_features = pickle.load(f)


FEATURE_TRANSLATIONS = {
    'Contract_Month-to-month': "Customer is on a short-term monthly contract",
    'Contract_Two year': "Customer is on a long-term contract (less likely to churn)",
    'InternetService_Fiber optic': "Using fiber optic internet (linked to higher churn)",
    'OnlineSecurity_No': "No online security service",
    'OnlineBackup_No': "No online backup enabled",
    'TechSupport_No': "No technical support included",
    'StreamingTV_Yes': "Subscribed to streaming TV",
    'StreamingMovies_Yes': "Subscribed to streaming movies",
    'PaymentMethod_Electronic check': "Pays via electronic check (often churns more)",
    'tenure': "Customer has been with company for short time",
    'MonthlyCharges': "Monthly charges are high",
    'TotalCharges': "Total charges are low (recent customer)",
}

RETENTION_TIPS = {
    'Contract_Month-to-month': "Encourage switching to yearly or two-year contracts",
    'Contract_Two year': "Reinforce loyalty benefits of long-term contracts",
    'InternetService_Fiber optic': "Emphasize the speed/value of fiber with loyalty rewards",
    'OnlineSecurity_No': "Offer a discount or free trial on security services",
    'OnlineBackup_No': "Recommend online backup as part of bundled packages",
    'TechSupport_No': "Promote 24/7 support availability or add-on services",
    'StreamingTV_Yes': "Bundle with loyalty entertainment offers",
    'StreamingMovies_Yes': "Remind customers of content value",
    'PaymentMethod_Electronic check': "Encourage switching to auto-pay or credit card",
    'tenure': "Provide early loyalty perks for new customers",
    'MonthlyCharges': "Offer a discount or better value package",
    'TotalCharges': "Send onboarding or loyalty messages to new customers",
}


def home(request):
    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                csv_file = request.FILES['csv_file']
                df = pd.read_csv(csv_file)

                original_ids = df.get("customerID", [f"Customer {i}" for i in range(len(df))])
                df.drop("customerID", axis=1, inplace=True, errors='ignore')

                # Fix TotalCharges
                if "TotalCharges" in df.columns:
                    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
                else:
                    df["TotalCharges"] = df.get("MonthlyCharges", 50) * df.get("tenure", 1)

                # Encode object columns
                for col in df.select_dtypes(include="object").columns:
                    df[col] = df[col].astype("category").cat.codes

                # Fill missing model features
                for col in model_features:
                    if col not in df.columns:
                        df[col] = 0

                df = df[model_features]

                # Predict
                probs = model.predict_proba(df)[:, 1] * 100


                # Get model coefficients
                coeffs = model.coef_[0]
                feature_contributions = df.values * coeffs  # shape: (n_samples, n_features)

                top_reasons = []
                for i in range(len(df)):
                    contrib = feature_contributions[i]
                    top_indices = contrib.argsort()[-3:][::-1]  # top 3
                    reasons = [model_features[idx] for idx in top_indices]
                    top_reasons.append(reasons)

                # Build predictions
                predictions = []

                for i in range(len(df)):
                    reasons_raw = top_reasons[i]

                    readable_reasons = []
                    tips = []

                    for reason in reasons_raw:
                        if isinstance(reason, str):
                            readable = FEATURE_TRANSLATIONS.get(reason, reason.replace("_", " ").capitalize())
                            tip = RETENTION_TIPS.get(reason, "Provide personalized support to retain the customer.")
                            readable_reasons.append(readable)
                            tips.append(tip)

                    predictions.append({
                        "customer_id": original_ids[i],
                        "churn_prob": round(probs[i], 2),
                        "reasons": readable_reasons,
                        "tips": tips
                    })
                # Count frequency of all reasons
                all_reasons = []
                for row in predictions:
                    all_reasons.extend(row['reasons'])

                reason_counts = Counter(all_reasons).most_common(5)  # top 5

                # Separate keys and values for chart.js
                reason_labels = [reason for reason, count in reason_counts]
                reason_values = [count for reason, count in reason_counts]

                # Churn risk summary for pie chart
                high_risk_count = sum(1 for p in predictions if p["churn_prob"] >= 50)
                low_risk_count = len(predictions) - high_risk_count
                risk_summary = {"high": high_risk_count, "low": low_risk_count}

                # Save for CSV download
                request.session['csv_results'] = predictions

                return render(request, 'churn_app/results.html', {
                    'predictions': predictions,
                    'risk_summary': risk_summary,
                    'reason_labels': reason_labels,
                    'reason_values': reason_values
                })

            except Exception as e:
                return render(request, 'churn_app/home.html', {
                    'form': form,
                    'error': f"Error processing file: {str(e)}"
                })
    else:
        form = UploadCSVForm()
    return render(request, 'churn_app/home.html', {'form': form})


# CSV download view
def download_csv(request):
    results = request.session.get('csv_results')

    if not results:
        return HttpResponse("No data to download", status=400)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="churn_predictions.csv"'

    writer = csv.writer(response)
    writer.writerow(['Customer ID', 'Churn Probability (%)', 'Top Reason 1', 'Top Reason 2', 'Top Reason 3', 'Tip 1', 'Tip 2', 'Tip 3'])

    for row in results:
        reasons = row.get("reasons", [])
        tips = row.get("tips", [])
        writer.writerow([
            row["customer_id"],
            row["churn_prob"],
            reasons[0] if len(reasons) > 0 else '',
            reasons[1] if len(reasons) > 1 else '',
            reasons[2] if len(reasons) > 2 else '',
            tips[0] if len(tips) > 0 else '',
            tips[1] if len(tips) > 1 else '',
            tips[2] if len(tips) > 2 else '',
        ])

    return response