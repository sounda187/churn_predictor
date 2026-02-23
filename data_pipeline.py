import polars as pl
import cerberus
import pandas as pd
import numpy as np
import os

print("🚀 Churn Predictor - GENERATING DATASET...")

class DataPipeline:
    def __init__(self):
        self.schema = {
            'MonthlyCharges': {'type': 'float', 'min': 0},
            'tenure': {'type': 'integer', 'min': 0},
            'Churn': {'type': 'integer', 'allowed': [0, 1]}
        }
        self.validator = cerberus.Validator(self.schema)
    
    def generate_data(self):
        """Create REALISTIC churn dataset (no internet needed!)"""
        print("📊 Generating 5000 customer records...")
        
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'customerID': [f'CUST_{i:05d}' for i in range(n_samples)],
            'tenure': np.random.randint(0, 73, n_samples),
            'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples),
            'TotalCharges': np.random.uniform(20.0, 9000.0, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])  # 27% churn rate
        }
        
        # Realistic churn logic
        for i in range(n_samples):
            if (data['tenure'][i] < 12 and 
                data['MonthlyCharges'][i] > 80 and 
                data['age'][i] > 50):
                data['Churn'][i] = 1
        
        df = pl.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        df.write_csv("data/raw_data.csv")
        
        print(f"✅ Generated {df.shape[0]} rows!")
        return df
    
    def clean_data(self, df):
        """Clean + feature engineering"""
        print("🧹 Cleaning & engineering features...")
        
        df = df.with_columns([
            (pl.col("tenure") / 12).alias("years_customer"),
            (pl.col("MonthlyCharges") * pl.col("tenure")).alias("total_spent"),
            (pl.col("TotalCharges") / pl.col("tenure").clip(1)).alias("avg_monthly")
        ]).drop_nulls()
        
        # Select features for ML
        df_clean = df.select([
            "customerID", "tenure", "MonthlyCharges", "TotalCharges", 
            "age", "years_customer", "total_spent", "avg_monthly", "Churn"
        ])
        
        df_clean.write_csv("data/cleaned_data.csv")
        print(f"✅ Cleaned: {df_clean.shape}")
        return df_clean
    
    def validate_data(self, df):
        """Simple validation"""
        print("🔍 Validating data quality...")
        sample = df.sample(min(50, df.height))
        records = sample.to_pandas().to_dict('records')
        
        valid = sum(1 for r in records if self.validator.validate(r))
        print(f"✅ {valid}/{len(records)} records valid!")
        return valid == len(records)

# RUN EVERYTHING
if __name__ == "__main__":
    pipeline = DataPipeline()
    raw_df = pipeline.generate_data()
    cleaned_df = pipeline.clean_data(raw_df)
    pipeline.validate_data(cleaned_df)
    print("\n🎉 DATA PIPELINE 100% COMPLETE!")
