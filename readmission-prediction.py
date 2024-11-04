import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

class ReadmissionPredictor:
    def __init__(self, data_path):
        """Initialize the Readmission Predictor."""
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess(self):
        """Load and preprocess the diabetes readmission dataset."""
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Create derived features
        self.df['total_diagnoses'] = self.df.apply(
            lambda x: sum(1 for i in range(1, 4) if pd.notna(x[f'diag_{i}'])), axis=1)
        
        self.df['multiple_medications'] = (self.df['num_medications'] > 5).astype(int)
        self.df['long_stay'] = (self.df['time_in_hospital'] > 7).astype(int)
        
        # Encode categorical variables
        categorical_columns = ['admission_type_id', 'discharge_disposition_id', 
                             'admission_source_id', 'medical_specialty']
        
        for column in categorical_columns:
            if column in self.df.columns:
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column].astype(str))
                self.label_encoders[column] = le
        
        # Handle missing values
        self.df = self.df.fillna(self.df.mean())
        
        return self
    
    def create_features(self):
        """Create and select features for the model."""
        # Select relevant features
        feature_columns = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses', 'total_diagnoses',
            'multiple_medications', 'long_stay', 'admission_type_id',
            'discharge_disposition_id', 'admission_source_id'
        ]
        
        self.X = self.df[feature_columns]
        self.y = (self.df['readmitted'] == '<30').astype(int)  # 1 if readmitted within 30 days
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        return self
    
    def train_model(self):
        """Train XGBoost model for readmission prediction."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Print performance metrics
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # Save model
        joblib.dump(self.model, 'readmission_model.joblib')
        
        return self
    
    def analyze_risk_factors(self):
        """Analyze and visualize key risk factors for readmission."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Feature importance
        plt.subplot(2, 2, 1)
        importance = pd.Series(self.model.feature_importances_, index=self.X.columns)
        sns.barplot(x=importance.values, y=importance.index)
        plt.title('Feature Importance in Predicting Readmission')
        
        # Plot 2: Readmission rate by length of stay
        plt.subplot(2, 2, 2)
        stays = pd.cut(self.df['time_in_hospital'], bins=5)
        readmit_by_stay = self.df.groupby(stays)['readmitted'].apply(
            lambda x: (x == '<30').mean() * 100)
        sns.barplot(x=readmit_by_stay.index.astype(str), y=readmit_by_stay.values)
        plt.title('Readmission Rate by Length of Stay')
        plt.xlabel('Days in Hospital')
        plt.ylabel('Readmission Rate (%)')
        
        # Plot 3: Readmission rate by number of medications
        plt.subplot(2, 2, 3)
        med_groups = pd.qcut(self.df['num_medications'], q=5)
        readmit_by_meds = self.df.groupby(med_groups)['readmitted'].apply(
            lambda x: (x == '<30').mean() * 100)
        sns.barplot(x=readmit_by_meds.index.astype(str), y=readmit_by_meds.values)
        plt.title('Readmission Rate by Number of Medications')
        plt.xlabel('Number of Medications (Quintiles)')
        plt.ylabel('Readmission Rate (%)')
        
        plt.tight_layout()
        plt.savefig('readmission_analysis.png')
        plt.close()
        
        return self

def main():
    """Main function to run the analysis."""
    predictor = ReadmissionPredictor('diabetic_data.csv')
    predictor.load_and_preprocess().create_features().train_model().analyze_risk_factors()
    
    print("\nAnalysis complete! Check readmission_analysis.png for visualizations.")

if __name__ == "__main__":
    main()
