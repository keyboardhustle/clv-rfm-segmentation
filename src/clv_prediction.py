import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

"""
Customer Lifetime Value (CLV) Prediction Model
================================================
Predicts 12-month CLV for B2B SaaS customers using
RFM features + product usage signals.

Approach:
- Feature engineering from transaction history
- Gradient Boosting regression model
- Outputs predicted CLV + customer tier classification

Usage:
    model = CLVPredictor()
    model.fit(transactions_df)
    predictions = model.predict(customers_df)
"""


class CLVPredictor:
    """
    Predicts 12-month Customer Lifetime Value using historical transaction data.
    
    Input DataFrame columns:
    - customer_id
    - transaction_date
    - revenue
    - product_tier (optional: 'free', 'starter', 'pro', 'enterprise')
    - num_users (optional: seat count)
    - support_tickets (optional: support volume)
    """

    CLV_TIERS = {
        'Platinum': (50000, float('inf')),
        'Gold': (20000, 50000),
        'Silver': (5000, 20000),
        'Bronze': (1000, 5000),
        'Low Value': (0, 1000)
    }

    def __init__(self, prediction_window_months: int = 12):
        self.prediction_window = prediction_window_months
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build predictive features from raw transaction history.
        Observation window: first 6 months of customer history.
        """
        df = df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])

        features = []

        for customer_id, group in df.groupby('customer_id'):
            group = group.sort_values('transaction_date')
            start_date = group['transaction_date'].min()
            obs_end = start_date + pd.DateOffset(months=6)

            obs = group[group['transaction_date'] <= obs_end]
            future = group[group['transaction_date'] > obs_end]

            if len(obs) == 0:
                continue

            # RFM features
            recency = (obs_end - obs['transaction_date'].max()).days
            frequency = len(obs)
            monetary = obs['revenue'].sum()
            avg_order = obs['revenue'].mean()
            std_order = obs['revenue'].std() if len(obs) > 1 else 0

            # Trend features
            first_half = obs[obs['transaction_date'] <= start_date + pd.DateOffset(months=3)]
            second_half = obs[obs['transaction_date'] > start_date + pd.DateOffset(months=3)]
            revenue_trend = (
                second_half['revenue'].sum() - first_half['revenue'].sum()
            ) / max(first_half['revenue'].sum(), 1)

            # Time between purchases
            if len(obs) > 1:
                time_deltas = obs['transaction_date'].diff().dt.days.dropna()
                avg_purchase_gap = time_deltas.mean()
            else:
                avg_purchase_gap = 180

            # Optional product signals
            product_tier_score = 0
            if 'product_tier' in obs.columns:
                tier_map = {'free': 0, 'starter': 1, 'pro': 2, 'enterprise': 3}
                product_tier_score = tier_map.get(obs['product_tier'].iloc[-1], 0)

            num_users = obs['num_users'].max() if 'num_users' in obs.columns else 1
            support_tickets = obs['support_tickets'].sum() if 'support_tickets' in obs.columns else 0

            # Target variable: actual revenue in next 12 months
            target_clv = future[
                future['transaction_date'] <= obs_end + pd.DateOffset(months=12)
            ]['revenue'].sum()

            features.append({
                'customer_id': customer_id,
                'recency_days': recency,
                'frequency': frequency,
                'monetary_6m': monetary,
                'avg_order_value': avg_order,
                'std_order_value': std_order,
                'revenue_trend_pct': revenue_trend,
                'avg_purchase_gap_days': avg_purchase_gap,
                'product_tier_score': product_tier_score,
                'num_users': num_users,
                'support_tickets': support_tickets,
                'clv_12m': target_clv
            })

        return pd.DataFrame(features)

    def fit(self, transactions_df: pd.DataFrame) -> 'CLVPredictor':
        """
        Train CLV prediction model on historical transaction data.
        """
        print("Engineering features...")
        feature_df = self._engineer_features(transactions_df)

        # Drop rows where we don't have enough future data
        feature_df = feature_df[feature_df['clv_12m'] > 0]
        print(f"Training on {len(feature_df)} customers with full observation windows")

        feature_cols = [
            'recency_days', 'frequency', 'monetary_6m', 'avg_order_value',
            'std_order_value', 'revenue_trend_pct', 'avg_purchase_gap_days',
            'product_tier_score', 'num_users', 'support_tickets'
        ]
        self.feature_names = feature_cols

        X = feature_df[feature_cols].fillna(0)
        y = feature_df['clv_12m']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance:")
        print(f"  MAE: ${mae:,.0f}")
        print(f"  R2 Score: {r2:.3f}")

        self.is_fitted = True
        return self

    def predict(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict 12-month CLV for all customers.
        Returns DataFrame with predicted CLV and tier classification.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        feature_df = self._engineer_features(transactions_df)
        X = feature_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)

        feature_df['predicted_clv_12m'] = self.model.predict(X_scaled)
        feature_df['predicted_clv_12m'] = feature_df['predicted_clv_12m'].clip(lower=0)

        # Assign tiers
        feature_df['clv_tier'] = feature_df['predicted_clv_12m'].apply(
            self._assign_tier
        )

        return feature_df[['customer_id', 'predicted_clv_12m', 'clv_tier',
                            'monetary_6m', 'frequency', 'recency_days']]

    def _assign_tier(self, clv_value: float) -> str:
        for tier, (low, high) in self.CLV_TIERS.items():
            if low <= clv_value < high:
                return tier
        return 'Low Value'

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from the trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        importances = self.model.feature_importances_
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)


if __name__ == '__main__':
    # Generate synthetic customer transaction data
    np.random.seed(42)

    customers = [f'C{i:04d}' for i in range(500)]
    records = []

    for cid in customers:
        # Each customer has 1-20 transactions over 2 years
        n_txns = np.random.randint(1, 21)
        start = pd.Timestamp('2022-01-01') + pd.Timedelta(days=np.random.randint(0, 180))
        tier = np.random.choice(['free', 'starter', 'pro', 'enterprise'], p=[0.3, 0.35, 0.25, 0.10])
        base_revenue = {'free': 0, 'starter': 200, 'pro': 800, 'enterprise': 3000}[tier]

        for _ in range(n_txns):
            start += pd.Timedelta(days=np.random.randint(15, 60))
            records.append({
                'customer_id': cid,
                'transaction_date': start,
                'revenue': base_revenue * (1 + np.random.normal(0, 0.2)),
                'product_tier': tier,
                'num_users': np.random.randint(1, 50),
                'support_tickets': np.random.randint(0, 5)
            })

    df = pd.DataFrame(records)
    df['revenue'] = df['revenue'].clip(lower=0)

    print("Training CLV Prediction Model")
    print(f"Dataset: {len(df)} transactions, {df['customer_id'].nunique()} customers\n")

    model = CLVPredictor(prediction_window_months=12)
    model.fit(df)

    print("\nFeature Importance:")
    print(model.feature_importance().to_string(index=False))

    predictions = model.predict(df)
    print("\nCLV Tier Distribution:")
    print(predictions['clv_tier'].value_counts())

    print("\nTop 10 Most Valuable Customers (Predicted):")
    print(predictions.nlargest(10, 'predicted_clv_12m')[[
        'customer_id', 'predicted_clv_12m', 'clv_tier', 'frequency'
    ]].to_string(index=False))
