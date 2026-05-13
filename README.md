# U.S. Macroeconomic Dashboard

Interactive dashboard showing U.S. unemployment and inflation trends from 1960 to present, including the Phillips Curve relationship by decade.

## Setup

**1. Get a free FRED API key**

Register at https://fred.stlouisfed.org/docs/api/api_key.html (takes 30 seconds).

**2. Add your API key**

Open `.env` and replace `your_api_key_here` with your actual key:

```
FRED_API_KEY=abc123yourkeyhere
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the dashboard**

```bash
python app.py
```

Open http://127.0.0.1:8050 in your browser.

## What you'll see

- **Time Series tab** — Unemployment rate and CPI inflation plotted from 1960 to present, with NBER recession bands shaded in the background.
- **Phillips Curve tab** — Scatter plot of unemployment vs. inflation, with each decade colored separately to show how the relationship has evolved (and broken down) over time.

## Data sources

- `UNRATE` — U.S. Unemployment Rate (Bureau of Labor Statistics via FRED)
- `CPIAUCSL` — Consumer Price Index for All Urban Consumers (BLS via FRED)
- Inflation is computed as the 12-month percentage change in CPI.
