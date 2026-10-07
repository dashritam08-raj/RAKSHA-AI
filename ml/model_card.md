# RAKSHA AI Risk Model Card

## Intended use
Decision-support for disaster operations. The ML score is a historical screening estimate, not a calibrated probability and not an autonomous emergency authority.

## Current model
**RAKSHA-ML-v1 — 70/30 Log-Target HGB + Shallow HGB**

Training period: ≤ 2011  
Final historical test period: 2012–2013

Historical test metrics:

- MAE: 2.8280
- RMSE: 4.6932
- R²: 0.1027
- Spearman: 0.4957

The final test period contains zero HIGH-impact events, and moderate/high-impact cases are strongly underpredicted. Therefore the model must not be presented as proven HIGH-risk detection or as a field-validated India model.

## Feature scope

The production inference contract uses 24 pre-event/contextual features covering:

- geography and coordinates
- district/province
- disaster type
- weather linkage
- temperature, precipitation and wind
- 7-day / 300-km earthquake context
- calendar seasonality
- data-availability flags

Post-event outcomes such as deaths, injured, damaged houses, economic loss and impact_score are explicitly forbidden as model inputs.

## Geography warning

The training data is Nepal historical data. Live coordinates outside the historical coordinate envelope are marked out-of-domain by the backend. Indian operational deployment requires India-representative training and validation.

## Output

- impact/risk screening score 0–100
- LOW / MODERATE / HIGH screening band
- component predictions
- blend weights
- data completeness
- input-quality metadata
- out-of-range warnings
- temporal/geographic applicability context

## Activation rule

The API status `validated` means the historical model-selection and chronological evaluation pipeline has been completed. It must not be interpreted as independent real-world or field validation.
