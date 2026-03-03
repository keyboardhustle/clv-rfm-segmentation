"""
RFM Scoring for B2B SaaS Customer Segmentation
===============================================
Problem: Identify high-value customer segments to prioritise CS and marketing spend.
Method:  Recency, Frequency, Monetary scoring + segment labelling.
Output:  rfm_scores.csv with segment per customer.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ──────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATA
# ──────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

n_customers = 2000
base_date = datetime(2026, 3, 1)

customers = []
for i in range(n_customers):
    plan = random.choices(['Starter', 'Growth', 'Enterprise'], weights=[0.5, 0.35, 0.15])[0]
    if plan == 'Enterprise':
        frequency = np.random.poisson(18)
        monetary = np.random.lognormal(mean=10.5, sigma=0.5)  # ~NOK 36k avg
        recency_days = np.random.randint(1, 45)
    elif plan == 'Growth':
        frequency = np.random.poisson(8)
        monetary = np.random.lognormal(mean=9.2, sigma=0.6)
        recency_days = np.random.randint(1, 120)
    else:
        frequency = np.random.poisson(3)
        monetary = np.random.lognormal(mean=7.8, sigma=0.8)
        recency_days = np.random.randint(1, 365)

    churned = random.random() < (0.05 if plan == 'Enterprise' else 0.2 if plan == 'Growth' else 0.4)
    if churned:
        recency_days = np.random.randint(90, 730)

    customers.append({
        'customer_id': f'CUST_{i+1:04d}',
        'plan_type': plan,
        'recency_days': max(1, int(recency_days)),
        'frequency': max(1, int(frequency)),
        'monetary_value': round(float(monetary), 2),
        'churned': churned,
        'last_purchase_date': (base_date - timedelta(days=recency_days)).strftime('%Y-%m-%d')
    })

df = pd.DataFrame(customers)
print(f"Dataset: {len(df)} customers | Plans: {df.plan_type.value_counts().to_dict()}")

# ──────────────────────────────────────────────
# 2. RFM SCORING (1-5 scale per dimension)
# ──────────────────────────────────────────────
# Recency: lower recency_days = higher score
df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

# Frequency: higher = better
df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

# Monetary: higher = better
df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

df['rfm_score'] = df['r_score'].astype(str) + df['f_score'].astype(str) + df['m_score'].astype(str)
df['rfm_total'] = df['r_score'] + df['f_score'] + df['m_score']

# ──────────────────────────────────────────────
# 3. SEGMENT LABELLING
# ──────────────────────────────────────────────
def assign_segment(row):
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal'
    elif r >= 3 and f <= 2:
        return 'Promising'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    elif r <= 2 and f >= 2:
        return 'Hibernating'
    else:
        return 'Lost'

df['segment'] = df.apply(assign_segment, axis=1)

# ──────────────────────────────────────────────
# 4. SEGMENT SUMMARY
# ──────────────────────────────────────────────
summary = df.groupby('segment').agg(
    customer_count=('customer_id', 'count'),
    avg_recency=('recency_days', 'mean'),
    avg_frequency=('frequency', 'mean'),
    avg_monetary=('monetary_value', 'mean'),
    total_revenue=('monetary_value', 'sum'),
    churn_rate=('churned', 'mean')
).round(1)

summary['pct_customers'] = (summary['customer_count'] / len(df) * 100).round(1)
summary['pct_revenue'] = (summary['total_revenue'] / df['monetary_value'].sum() * 100).round(1)
summary = summary.sort_values('total_revenue', ascending=False)

print("\n=== RFM SEGMENT SUMMARY ===")
print(summary[['customer_count', 'pct_customers', 'pct_revenue', 'avg_monetary', 'churn_rate']].to_string())

# ──────────────────────────────────────────────
# 5. BUSINESS RECOMMENDATIONS OUTPUT
# ──────────────────────────────────────────────
champions = summary.loc['Champions'] if 'Champions' in summary.index else None
at_risk = summary.loc['At Risk'] if 'At Risk' in summary.index else None

print("\n=== RECOMMENDATIONS ===")
if champions is not None:
    print(f"Champions ({champions['pct_customers']}% of customers) drive {champions['pct_revenue']}% of revenue.")
    print("  -> Prioritise dedicated CSM, QBRs, and upsell conversations.")
if at_risk is not None:
    print(f"At Risk ({at_risk['pct_customers']}% of customers): high historical value, declining recency.")
    print("  -> Trigger proactive re-engagement within 14 days. Do not wait for churn.")
print("Lost segment: win-back rate <3%. Reduce email spend on this group.")

# Save output
df.to_csv('data/rfm_scored_customers.csv', index=False)
print("\nScored dataset saved to data/rfm_scored_customers.csv")
