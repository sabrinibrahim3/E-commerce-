# BI_project_FINAL0 Notebook Review

## Overview
This notebook explores an e-commerce order dataset from Kaggle, covering orders, items, customers, payments, and products across train/test splits. It performs data loading, cleaning, feature engineering, modeling (XGBoost regression), RFM segmentation, and Prophet-based forecasting, finishing by exporting cleaned tables.

## Data Ingestion and Structure
- Imports common analytics libraries and configures pandas display options before loading CSVs from a Google Drive mount into train/test DataFrames for each table. Shapes, column counts, and memory footprints are printed to verify completeness (89,316 training rows; 38,279 test rows).【F:notebook_clean.md†L68-L159】【F:notebook_clean.md†L176-L205】
- Key relationships are documented: `order_id` links orders to items/payments, `product_id` links products to order items, and `customer_id` links orders to customers.【F:notebook_clean.md†L231-L238】

## Data Cleaning
- Missing values are profiled for every table in both splits.【F:notebook_clean.md†L254-L279】
- Timestamps are coerced to datetime; missing approval timestamps default to purchase time. Product categories fall back to `unknown`, and numeric product dimensions are imputed with training medians to maintain consistency across train/test.【F:notebook_clean.md†L301-L333】
- Categorical and identifier columns are explicitly typed as strings/categories across entities (orders, order items, customers, payments, products) to stabilize merges and modeling.【F:notebook_clean.md†L335-L368】
- Delivery timestamps with 1,889 nulls are filled using estimated delivery dates to avoid losing records.【F:notebook_clean.md†L386-L399】
- Duplicate checks highlight heavy duplication in product IDs (~70%), followed by deduplication of products separately in train/test.【F:notebook_clean.md†L406-L485】
- Extensive outlier handling removes extreme prices, shipping charges, and records with excessive shipping-to-price ratios, then cascades deletions across dependent tables to keep referential integrity.【F:notebook_clean.md†L488-L724】

## Feature Engineering
- Builds a delivery target (`delivery_days`) by subtracting purchase from delivered timestamps, using estimated dates where necessary, and drops negative or missing targets.【F:notebook_clean.md†L743-L783】
- Creates an order-level modeling frame by aggregating payment value, product weights, item counts, and average dimensions, plus temporal features (purchase day of week/hour) and order status encoding.【F:notebook_clean.md†L791-L835】

## Modeling and Analytics
- Trains an XGBoost regressor on engineered features; performance reaches MAE ≈ 6.1 days and RMSE ≈ 8.9 days. Feature importances are reported to gauge drivers of delivery time.【F:notebook_clean.md†L849-L945】
- Performs RFM analysis on customer orders, generates pivot-based heatmaps, and segments customers into At Risk, Potential, and Champions categories, with interpretive guidance for the heatmap.【F:notebook_clean.md†L117-133】【F:notebook_clean.md†L131-L1183】
- Aggregates daily revenue, removes outliers via IQR, and fits a Prophet model with weekly/yearly seasonality to forecast 90 days ahead, including interpretation of trend and seasonal components.【F:notebook_clean.md†L1204-L1319】

## Visualization and Insights
- Explores distributions for product weights, payment values, order status counts, payment value by method, and correlation heatmaps, supplying narrative insights (e.g., right-skewed payment distributions and limited correlation between delivery time and other features).【F:notebook_clean.md†L1339-L1490】【F:notebook_clean.md†L1426-L1506】

## Outputs
- Exports cleaned train and test tables for orders, order items, products, payments, and customers to CSV files for downstream use.【F:notebook_clean.md†L1510-L1533】

## Strengths
- Comprehensive coverage from ingestion through modeling and forecasting in a single notebook.
- Consistent handling of missing values and data types across train/test splits.
- Careful cascade deletions after outlier filtering to preserve referential integrity.
- Clear interpretive markdown accompanying modeling outputs and business-oriented insights.

## Risks and Recommendations
- Strong reliance on hard-coded Google Drive paths reduces portability; parameterize input/output paths for reuse.
- Outlier thresholds (e.g., price > 6,000; shipping_price_ratio > 5; shipping_charges > 200) are heuristic—validate them with domain input or percentile-based rules to avoid biasing results.【F:notebook_clean.md†L488-L724】
- Product deduplication notes advise keeping train/test separate; ensure downstream merges reference the deduplicated copies to prevent reintroduction of duplicates.【F:notebook_clean.md†L476-L485】
- XGBoost training omits cross-validation and early stopping; consider hyperparameter tuning and time-based validation splits to better reflect delivery-time forecasting challenges.【F:notebook_clean.md†L866-L945】
- Prophet modeling assumes additive seasonality and limited changepoint flexibility; evaluate parameter sensitivity and include backtesting to quantify forecast reliability.【F:notebook_clean.md†L1204-L1319】
- Saving cleaned datasets occurs at the end of the notebook; add lightweight data quality assertions (row counts, duplicate checks) post-export to verify integrity.【F:notebook_clean.md†L1510-L1533】
