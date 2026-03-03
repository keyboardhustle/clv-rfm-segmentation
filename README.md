# CLV & RFM Customer Segmentation

## Problem

A B2B SaaS company with 2,000+ customers is spending its CRM and retention budget evenly across all accounts. Customer Success is stretched thin. Marketing is running the same re-engagement campaigns to churned free users and churned paying enterprise customers. The question: **who are our most valuable customers, and how do we prioritise budget around them?**

## Approach

This project applies two complementary frameworks:

1. **RFM Scoring** (Recency, Frequency, Monetary) — rule-based segmentation that ranks every customer across three dimensions and assigns a segment label (Champions, Loyal, At Risk, Lost, etc.).
2. **BG/NBD + Gamma-Gamma CLV Model** — probabilistic model that estimates each customer’s expected future value over a 12-month horizon, used to rank accounts by predicted revenue.
3. **K-means Clustering** — unsupervised clustering on RFM features to validate and refine the segments without relying on manual thresholds.

## Data

Synthetic dataset of 2,000 B2B SaaS customers generated to reflect realistic distributions:

| Field | Description |
|-------|-------------|
| `customer_id` | Unique customer ID |
| `first_purchase_date` | Account creation / first payment |
| `last_purchase_date` | Most recent transaction |
| `frequency` | Number of transactions in observation window |
| `monetary_value` | Average transaction value (NOK) |
| `plan_type` | Starter / Growth / Enterprise |
| `churned` | Boolean flag |

See `data/customers_synthetic.csv`.

## Key Files

```
/data
  customers_synthetic.csv        → synthetic customer dataset
/notebooks
  01_rfm_scoring.ipynb           → RFM calculation, scoring, segment labelling
  02_clv_bgnbd_model.ipynb       → BG/NBD + Gamma-Gamma CLV modelling
  03_kmeans_clustering.ipynb     → K-means cluster validation
  04_segment_profiles.ipynb      → Segment deep-dives and visualisations
/outputs
  rfm_segments_summary.png       → Segment size and value chart
  clv_distribution.png           → CLV distribution by segment
  cluster_plot.png               → 2D cluster visualisation
README.md
```

## Key Findings

| Segment | % of Customers | % of Revenue | Avg CLV (12m) |
|---------|---------------|-------------|---------------|
| Champions | 8% | 41% | NOK 42,000 |
| Loyal | 14% | 28% | NOK 18,500 |
| At Risk | 19% | 16% | NOK 7,200 |
| Hibernating | 23% | 9% | NOK 2,100 |
| Lost | 36% | 6% | NOK 800 |

- Top 22% of customers generate 69% of revenue.
- “At Risk” segment has the highest intervention ROI: high historical value but declining recency.
- K-means validation confirmed RFM thresholds were correctly placed (silhouette score: 0.61).

## Business Recommendations

1. **CS focus:** Assign dedicated CSM capacity to Champions + Loyal (22% of accounts). Flag all “At Risk” accounts for proactive QBR outreach within 30 days.
2. **CRM budget reallocation:** Shift 60% of retention campaign budget away from “Lost” toward “At Risk”. Lost-segment win-back rate historically under 3% — not worth the spend.
3. **Upsell targeting:** Use CLV scores to prioritise upsell motions. Accounts in top CLV decile with Starter or Growth plans are highest-probability upsell candidates.
4. **Marketing segmentation:** Stop sending re-engagement emails to all churned accounts. Segment by CLV and only re-engage accounts with predicted 12m CLV above NOK 5,000.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn lifetimes scikit-learn
jupyter notebook notebooks/01_rfm_scoring.ipynb
```

## Stack

- Python 3.11
- pandas, numpy
- lifetimes (BG/NBD, Gamma-Gamma)
- scikit-learn (KMeans)
- matplotlib, seaborn
