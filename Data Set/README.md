# Dataset

The raw e-commerce dataset is **excluded from this repository** because the
CSV (~391 MB) and XLSX (~359 MB) exceed GitHub's 100 MB per-file limit.

## Expected file

```
Data Set/ecommerce_dataset_+1m.csv      # 1,000,123 rows x 60 columns
```

## How to obtain it

Place the CSV in this folder. You can use any of:

1. **Original source** — the file you received as part of the assignment
   (Big Data Analytics mini project).
2. **Shared drive / submission archive** — the project author distributes it
   alongside the demo video and slide deck.
3. **Regenerate** — the CSV was generated synthetically; if you have access
   to the generator notebook used by the course, re-run it.

## Schema (60 columns)

| Group | Columns |
|---|---|
| Order | `order_id`, `order_date`, `order_year/month/day/hour/minute/second`, `is_weekend`, `order_status`, `return_reason` |
| Customer | `customer_id`, `customer_name`, `gender`, `age`, `customer_segment`, `country`, `city`, `customer_loyalty_score`, `total_orders_by_customer`, `account_creation_date` |
| Product | `product_id`, `product_name`, `category`, `sub_category`, `brand`, `product_rating_avg`, `product_reviews_count`, `stock_quantity` |
| Pricing | `unit_price_usd`, `quantity`, `discount_percent`, `discount_amount_usd`, `total_price_usd`, `cost_usd`, `profit_usd`, `tax_usd`, `currency` |
| Payment | `payment_method`, `payment_status`, `installment_plan` |
| Shipping | `shipping_method`, `shipping_cost_usd`, `delivery_days`, `shipping_country`, `warehouse_location`, `delivery_status` |
| Reviews | `rating`, `review_sentiment`, `customer_feedback` |
| Marketing | `coupon_used`, `coupon_code`, `campaign_source` |
| Web analytics | `device_type`, `traffic_source`, `session_duration_minutes`, `pages_visited`, `abandoned_cart_before` |
| Risk / Ops | `fraud_risk_score`, `profit_margin_percent`, `order_priority`, `support_ticket_created` |

Once the CSV is in this folder, run:

```powershell
python run_pipeline.py            # pandas + sklearn pipeline
python run_spark_pipeline.py      # PySpark + Spark MLlib + ALS pipeline
```
