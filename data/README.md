# Data Directory

## Structure

- raw/ — Original generated dataset (customer_support_dataset.csv)
- processed/ — Cleaned and preprocessed data
- external/ — Any external datasets

## Dataset Info

- Total samples: 1800
- Intent classes: 9
- Samples per class: 200 (balanced)
- Features: text, intent, priority, department

## Generating Data

```bash
py data/generate_dataset.py
```

## Intent Classes

| Intent | Priority | Department |
|---|---|---|
| payment_issue | critical | Billing Team |
| account_locked | critical | Security Team |
| refund_request | high | Billing Team |
| technical_bug | high | Engineering Team |
| subscription_cancel | high | Retention Team |
| invoice_problem | medium | Finance Team |
| shipping_delay | medium | Logistics Team |
| feature_request | low | Product Team |
| general_inquiry | low | General Support |
