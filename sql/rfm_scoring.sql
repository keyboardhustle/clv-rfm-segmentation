-- ============================================================
-- RFM Scoring and Customer Segmentation SQL
-- ============================================================
-- Purpose: Score all customers on Recency, Frequency, Monetary
--          and assign segments for CRM and marketing targeting.
-- Database: Works with PostgreSQL, BigQuery, Snowflake
-- Input table: orders (customer_id, order_date, revenue)
-- ============================================================

-- Step 1: Calculate raw RFM values per customer
WITH rfm_raw AS (
    SELECT
        customer_id,
        MAX(order_date)                          AS last_order_date,
        COUNT(DISTINCT order_id)                 AS frequency,
        SUM(revenue)                             AS monetary,
        DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) AS recency_days
    FROM `your_project.your_dataset.orders`
    WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
      AND status = 'completed'
    GROUP BY customer_id
),

-- Step 2: Assign quintile scores (1-5) for each dimension
-- 5 = best (most recent, most frequent, highest spend)
rfm_scores AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        last_order_date,

        -- Recency: lower days = better = higher score
        NTILE(5) OVER (ORDER BY recency_days DESC)  AS r_score,

        -- Frequency: higher = better = higher score
        NTILE(5) OVER (ORDER BY frequency ASC)      AS f_score,

        -- Monetary: higher = better = higher score
        NTILE(5) OVER (ORDER BY monetary ASC)       AS m_score

    FROM rfm_raw
),

-- Step 3: Combine scores into single RFM score
rfm_combined AS (
    SELECT
        *,
        CAST(r_score AS STRING) || CAST(f_score AS STRING) || CAST(m_score AS STRING) AS rfm_cell,
        (r_score + f_score + m_score) AS rfm_total_score,
        ROUND((r_score * 0.4 + f_score * 0.3 + m_score * 0.3), 2) AS rfm_weighted_score
    FROM rfm_scores
),

-- Step 4: Assign named segments
rfm_segmented AS (
    SELECT
        *,
        CASE
            -- Champions: bought recently, buy often, spend most
            WHEN r_score = 5 AND f_score >= 4 AND m_score >= 4
                THEN 'Champions'

            -- Loyal Customers: buy regularly, decent recency
            WHEN f_score >= 4 AND r_score >= 3
                THEN 'Loyal Customers'

            -- Potential Loyalists: recent buyers with average frequency
            WHEN r_score >= 4 AND f_score BETWEEN 2 AND 3
                THEN 'Potential Loyalists'

            -- Recent Customers: bought recently but not often
            WHEN r_score = 5 AND f_score <= 2
                THEN 'New Customers'

            -- Promising: recent, some engagement
            WHEN r_score >= 4 AND f_score = 1
                THEN 'Promising'

            -- Need Attention: above average but not engaged recently
            WHEN r_score = 3 AND f_score >= 3 AND m_score >= 3
                THEN 'Need Attention'

            -- About to Sleep: below average recency, frequency, monetary
            WHEN r_score = 3 AND f_score <= 2
                THEN 'About to Sleep'

            -- At Risk: spent big and purchased often but not recently
            WHEN r_score <= 2 AND f_score >= 4 AND m_score >= 4
                THEN 'At Risk'

            -- Cannot Lose: made huge purchases, but not recently
            WHEN r_score = 1 AND f_score >= 4
                THEN 'Cannot Lose Them'

            -- Hibernating: last purchase was long ago, low frequency
            WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2
                THEN 'Hibernating'

            -- Lost Customers: lowest scores
            WHEN r_score = 1 AND f_score = 1 AND m_score = 1
                THEN 'Lost'

            ELSE 'Others'
        END AS rfm_segment
    FROM rfm_combined
)

-- Final output with segment summary
SELECT
    customer_id,
    last_order_date,
    recency_days,
    frequency,
    ROUND(monetary, 2)      AS monetary,
    r_score,
    f_score,
    m_score,
    rfm_cell,
    rfm_total_score,
    rfm_weighted_score,
    rfm_segment
FROM rfm_segmented
ORDER BY rfm_weighted_score DESC;


-- ============================================================
-- Segment Summary Report
-- ============================================================
-- Run this separately to get aggregate segment metrics

SELECT
    rfm_segment,
    COUNT(*)                                AS customer_count,
    ROUND(AVG(monetary), 0)                 AS avg_ltv,
    ROUND(SUM(monetary), 0)                 AS total_revenue,
    ROUND(AVG(recency_days), 0)             AS avg_recency_days,
    ROUND(AVG(frequency), 1)                AS avg_frequency,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct_of_customers,
    ROUND(SUM(monetary) * 100.0 / SUM(SUM(monetary)) OVER (), 1) AS pct_of_revenue
FROM rfm_segmented
GROUP BY rfm_segment
ORDER BY total_revenue DESC;
