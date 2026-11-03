# databricks_bank_lakehouse_360

Banking 360 lakehouse demo — landing → Bronze → Silver → Gold with dbt, Unity Catalog, Genie, AI/BI Dashboard, and a product-owner chatbot.

## Structure
- `data_generator/` — synthetic retail + wholesale banking data (Parquet)
- `tests/` — data-contract validator
- `landing/` — generated parquet (gitignored)

## Runbook
1. `python data_generator/generate_banking_data.py`
2. `python tests/validate_landing.py`
