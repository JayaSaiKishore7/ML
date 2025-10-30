"""
Generate sample customer churn data for demonstration purposes.
This script creates a synthetic dataset similar to the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)


def generate_sample_data(n_samples=1000):
    """Generate sample customer data."""
    
    print(f"Generating {n_samples} sample customer records...")
    
    # Customer tenure (months)
    tenure = np.random.randint(0, 73, n_samples)
    
    # Monthly charges
    monthly_charges = np.random.uniform(18, 120, n_samples)
    
    # Total charges (based on tenure and monthly charges with some noise)
    total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_samples)
    total_charges = np.maximum(total_charges, 0)
    
    # Contract type
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract = np.random.choice(contract_types, n_samples, p=[0.55, 0.25, 0.20])
    
    # Payment method
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    payment_method = np.random.choice(payment_methods, n_samples, p=[0.35, 0.20, 0.25, 0.20])
    
    # Internet service
    internet_services = ['DSL', 'Fiber optic', 'No']
    internet_service = np.random.choice(internet_services, n_samples, p=[0.35, 0.45, 0.20])
    
    # Online security
    online_security = np.where(
        internet_service == 'No',
        'No internet service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])
    )
    
    # Tech support
    tech_support = np.where(
        internet_service == 'No',
        'No internet service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    )
    
    # Gender
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Senior citizen
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    
    # Partner
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    
    # Dependents
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # Phone service
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    # Multiple lines
    multiple_lines = np.where(
        phone_service == 'No',
        'No phone service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.42, 0.58])
    )
    
    # Paperless billing
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    
    # Generate churn based on features (with some logic)
    churn_prob = 0.1  # Base churn probability
    
    # Increase churn probability based on factors
    churn_prob = np.where(contract == 'Month-to-month', churn_prob + 0.3, churn_prob)
    churn_prob = np.where(payment_method == 'Electronic check', churn_prob + 0.15, churn_prob)
    churn_prob = np.where(tenure < 12, churn_prob + 0.2, churn_prob)
    churn_prob = np.where(monthly_charges > 80, churn_prob + 0.1, churn_prob)
    churn_prob = np.where(tech_support == 'No', churn_prob + 0.1, churn_prob)
    churn_prob = np.where(online_security == 'No', churn_prob + 0.1, churn_prob)
    
    # Decrease churn probability based on factors
    churn_prob = np.where(contract == 'Two year', churn_prob - 0.15, churn_prob)
    churn_prob = np.where(tenure > 48, churn_prob - 0.2, churn_prob)
    
    # Clip probabilities
    churn_prob = np.clip(churn_prob, 0, 0.9)
    
    # Generate churn labels
    churn = np.random.binomial(1, churn_prob)
    churn = np.where(churn == 1, 'Yes', 'No')
    
    # Create dataframe
    df = pd.DataFrame({
        'customerID': [f'CUST{i:05d}' for i in range(n_samples)],
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'Churn': churn
    })
    
    return df


def main():
    """Generate and save sample data."""
    # Generate data
    df = generate_sample_data(n_samples=2000)
    
    # Create directories
    raw_dir = os.path.join(os.path.dirname(__file__), 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # Save data
    output_path = os.path.join(raw_dir, 'customer_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\nSample data saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"\nChurn distribution:")
    print(df['Churn'].value_counts())
    print(f"\nChurn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")


if __name__ == '__main__':
    main()
