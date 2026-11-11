# JPMorgan Chase (JPM): Stylized Facts of Returns & Factor Models

Empirical analysis of JPMorgan Chase (NYSE: JPM) stock returns from 1980 to 2025, carried out in Python as the midterm project of the **Financial Econometrics with Python** course at **EDHEC Business School** (2026–2027).

- **Part 1:** stylized facts of financial returns, tested at daily, weekly, monthly and annual frequencies.
- **Part 2:** CAPM and Fama-French three-factor regressions on monthly excess returns.

---

## Part 1: Stylized facts of JPM returns

**Sample:** JPM daily closing prices, 17 March 1980 to 31 December 2025 (11,543 observations, 45.8 years). Log-returns at daily, weekly, monthly and annual frequencies.

| Topic | Methods |
|---|---|
| Summary statistics | Moments, quantiles, Jarque-Bera and Lilliefors tests, by frequency, by sub-period (4 equal blocks) and with/without the October 1987 crash |
| Stationarity | Price vs. log-price vs. return; ACF of log-prices; rolling mean and standard deviation (252 days) with confidence bands; block moments with linear trend; Cochran's Q homogeneity test |
| Asymmetry | Skewness as a function of the sampling horizon (h = 1 to 252 days), one-sided z-test, crossing horizon |
| Heavy tails | Kernel density vs. normal, Student-t QQ plots (ν = 10, 5, 3), kurtosis as a function of the horizon, one-sided z-test |
| Aggregational Gaussianity | Jarque-Bera, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests by frequency |
| No autocorrelation of returns | ACF with Bartlett bands, Box-Pierce and Ljung-Box tests |
| Volatility clustering | ACF of squared and absolute returns, Ljung-Box test on squared returns |
| Leverage effect | Cross-correlations corr(r_{t−j}, r_t²) with joint test (1988–2025); regression of daily VIX changes on JPM returns; asymmetric response (F-test, H0: β⁻ = β⁺) |

## Part 2: Factor models

**Sample:** monthly excess log-returns, April 1980 to December 2025 (T = 549). Market factor = CRSP value-weighted return (Mkt-RF + RF).

- **Model 0:** constant only
- **Model 1:** CAPM
- **Model 2:** Fama-French three factors (MKT, SMB, HML)
- OLS vs. White heteroskedasticity-consistent standard errors (HC0, HC3)

## Selected results

| | Model 0 | CAPM | FF3 |
|---|---|---|---|
| α (% per month) | 0.713* | −0.244 | −0.521* |
| β MKT | | 1.272*** | 1.378*** |
| β SMB | | | 0.172* |
| β HML | | | 0.872*** |
| Adj. R² | 0.000 | 0.393 | 0.477 |

OLS standard errors; * p < 0.10, *** p < 0.01.

---

## Data

- **JPM and VIX (^VIX) daily prices** are downloaded at runtime from Yahoo Finance via `yfinance` (internet connection required).
- **Fama-French factors:** `FF_Research_Data_Factors_monthly.csv`, the monthly block of the *Fama/French 3 Factors* file from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). It must be in the same folder as the notebook.

Yahoo Finance may revise historical data, so results can differ slightly from those reported above.

## Repository structure

```
├── JPM_financial_econometrics.ipynb
├── FF_Research_Data_Factors_monthly.csv
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/tomrousselet/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook JPM_financial_econometrics.ipynb
```

Requires Python 3.10+ (developed with Python 3.14).

## Author

**Tom Rousselet**, EDHEC Business School & École Centrale de Lille
