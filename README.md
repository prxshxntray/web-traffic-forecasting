This notebook implements a robust baseline for the Kaggle
Web Traffic Time Series Forecasting dataset.

## Key ideas
- log1p transform for heavy-tailed traffic
- median-based level estimation
- weekday seasonality
- deterministic deduplication of overlapping keys

## Structure
1. Utilities
2. Baseline model
3. Visual intuition
4. Time-based validation
5. Submission generation

This notebook is written as a report-style pipeline.
