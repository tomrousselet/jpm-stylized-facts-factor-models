# Stylized Facts and Factor Exposures of JPMorgan Chase Stock Returns (1980–2025)

This is an empirical study, written in Python, of 45 years of JPMorgan Chase & Co. (NYSE: JPM) stock returns. It has two parts.

- **Part I** tests seven classical stylized facts of financial returns at daily, weekly, monthly and annual frequencies. Standard tests assume i.i.d. normal returns, so each one is checked against a procedure that stays valid under heavy tails and conditional heteroskedasticity: White standard errors, Hill estimator, ARCH-LM test, GARCH(1,1) by maximum likelihood.
- **Part II** measures JPM's systematic risk exposures with the CAPM and the Fama-French three-factor model, using OLS and White inference.

The complete analysis, with all code, tables and figures, is in [`JPM_financial_econometrics.ipynb`](JPM_financial_econometrics.ipynb).

---

## Key findings

| Stylized fact | Verdict | Main evidence |
|---|---|---|
| Stationarity of returns |  Mean and autocovariance; variance compatible | Block homogeneity test: p = 0.90 (mean), p = 0.25 (autocovariance). The variance test rejects (Q/df = 8.8) because of volatility clustering, but GARCH persistence is below 1 |
| Negative skewness |  Weekly to monthly horizons only | Daily skewness of −0.11 becomes +0.17 once the October 1987 crash is removed; −0.34 at 5 days and −0.53 at 21 days |
| Heavy tails |  Strong | Daily kurtosis 18.3; tail index 2.5–3.0 (Hill) and Student-t ν̂ = 2.8 (maximum likelihood) |
| Aggregational Gaussianity |  As a tendency | Kurtosis falls from 18.3 (daily) to 4.9 (annual), JB/N from 9.74 to 0.31; normality still rejected at one year |
| No linear autocorrelation |  Daily and weekly, after correcting for heteroskedasticity | Ljung-Box rejects, but the White-robust F-test does not (p = 0.43 daily, 0.46 weekly) |
| Volatility clustering |  Strong | ARCH-LM = 1,581 (25 lags); GARCH(1,1) persistence 0.993, half-life of a shock 93 trading days |
| Leverage effect |  Right sign; borderline for JPM alone, clear at market level | 9 of 10 lagged coefficients negative, robust F-test p = 0.047 to 0.091; the VIX reacts 1.8× more to JPM falls than to JPM rises |

| Factor model | Result |
|---|---|
| Market beta | 1.24 (CAPM), 1.35 (Fama-French); significantly above 1 |
| Value exposure | HML loading 0.87 (t = 9.5): JPM behaves like a value stock |
| Size exposure | Not significant |
| Alpha | Not significant: +0.19% per month (CAPM), −0.09% per month (Fama-French) |
| Explained variance | R² = 0.38 (CAPM), 0.47 (Fama-French) |

---

## Contents

1. [Data](#1-data)
2. [Methodology](#2-methodology)
3. [Part I: Stylized facts of returns](#3-part-i-stylized-facts-of-returns)
4. [Part II: Factor models](#4-part-ii-factor-models)
5. [Limitations](#5-limitations)
6. [Reproducing the results](#6-reproducing-the-results)

---

## 1. Data

| Series | Source | Frequency | Sample |
|---|---|---|---|
| JPM adjusted closing price | Yahoo Finance (`yfinance`) | Daily | 17 March 1980 to 31 December 2025, 11,543 prices |
| CBOE Volatility Index (^VIX) | Yahoo Finance (`yfinance`) | Daily | January 1990 to December 2025 |
| Mkt-RF, SMB, HML, RF | [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Monthly | April 1980 to December 2025, 549 months |

**Part I** uses log-returns $r_t = \ln P_t - \ln P_{t-1}$ at four frequencies. Weekly, monthly and annual prices are the last observations of each calendar period. This gives 11,542 daily, 2,389 weekly, 549 monthly and 45 annual returns.

**Part II** uses monthly simple excess returns (see [section 4.5](#45-simple-versus-log-returns)).

**The October 1987 crash.** The largest daily loss in the sample, a log-return of −32.5%, occurred during the week of 19 October 1987. Every result sensitive to extreme observations is therefore re-estimated without the five trading days from 19 to 23 October 1987.

---

## 2. Methodology

The textbook tests for stylized facts are calibrated under i.i.d. normal returns: the skewness test $z = \hat S/\sqrt{6/N}$, the kurtosis test $z = (\hat K - 3)/\sqrt{24/N}$, the Bartlett bands $\pm 1.96/\sqrt{N}$ and the Box-Pierce and Ljung-Box statistics. Equity returns are heavy-tailed and conditionally heteroskedastic. Under those conditions, the variance of a sample autocorrelation exceeds $1/N$, and the tests over-reject. Each standard test is therefore paired with a complement that stays valid in that setting.

| Question | Standard test | Robust complement |
|---|---|---|
| Are the moments constant over time? | Rolling moments, linear trend on block moments | Homogeneity test across non-overlapping blocks |
| How heavy are the tails? | Kurtosis z-test, kurtosis-implied Student-t ν | Hill estimator, Student-t maximum likelihood |
| Do returns converge to normality? | Jarque-Bera, Lilliefors, Anderson-Darling | JB/N, a shape measure comparable across sample sizes |
| Are returns autocorrelated? | Bartlett bands, Ljung-Box | Autoregression with White standard errors |
| Is volatility clustered? | ACF of $r_t^2$ and $\lvert r_t\rvert$, Ljung-Box on $r_t^2$ | Engle's ARCH-LM test, GARCH(1,1) by maximum likelihood |
| Is there a leverage effect? | $\text{corr}(r_{t-j}, r_t^2)$ with $z = \hat\rho\sqrt{N}$ | Regression of $r_t^2$ on lagged returns with White standard errors; VIX regressions |

All tests are run at the 5% level. White's heteroskedasticity-consistent covariance matrix is

$$\widehat{\text{Var}}_{\text{White}}(\hat\beta) = (X'X)^{-1}\Big(\sum_t \hat u_t^2\, x_t x_t'\Big)(X'X)^{-1},$$

which weights each observation by its own squared residual instead of assuming one common variance.

---

## 3. Part I: Stylized facts of returns

### 3.1 Summary statistics

**Table 1.** JPM log-returns. Moments and extremes in percent. The ex-crash column removes 19–23 October 1987.

| | Daily | Daily, ex-crash | Weekly | Monthly | Annual |
|---|---:|---:|---:|---:|---:|
| Mean (%) | 0.050 | 0.052 | 0.238 | 1.040 | 12.238 |
| Standard deviation (%) | 2.205 | 2.182 | 4.746 | 9.036 | 30.422 |
| Skewness | −0.106 | 0.166 | −0.210 | −0.743 | −0.964 |
| Kurtosis | 18.29 | 14.76 | 11.81 | 5.55 | 4.90 |
| Minimum (%) | −32.46 | −23.23 | −41.68 | −42.58 | −92.36 |
| 5% quantile (%) | −3.15 | −3.15 | −7.28 | −13.94 | −35.21 |
| Median (%) | 0.00 | 0.00 | 0.30 | 1.41 | 11.05 |
| 95% quantile (%) | 3.24 | 3.24 | 7.26 | 14.78 | 51.36 |
| Maximum (%) | 22.39 | 22.39 | 39.94 | 28.45 | 73.04 |
| Jarque-Bera | 112,450 | 66,481 | 7,739 | 199.3 | 13.75 |
| Observations | 11,542 | 11,537 | 2,389 | 549 | 45 |

The mean daily return of 0.050% is significantly positive: its 95% confidence interval is [0.010%, 0.090%].

### 3.2 Fact 1: Returns are stationary

![JPM price, log-price and log-return](figures/price_logprice_returns.png)

*Figure 1. Daily price, log-price and log-return, with the fitted trends $P_0e^{\hat\mu t}$, $p_0+\hat\mu t$ and $\hat\mu$ (red).*

**Prices versus returns.** Price and log-price trend upwards. The autocorrelation of the daily log-price is still 0.90 at lag 250, fifty times the Bartlett half-width of 0.018. This very slow decay is the signature of a non-stationary series. Returns fluctuate around a constant level, and their autocorrelations stay inside or at the edge of the band at every lag.

![Rolling mean and standard deviation](figures/rolling_mean_std.png)

*Figure 2. Rolling mean and standard deviation of daily log-returns over 252 days, with 95% confidence bands (red).*

The rolling mean oscillates around its full-sample value, and its confidence band almost always contains it. The rolling standard deviation, by contrast, moves from about 1% to almost 7% during 2008–2009, far outside its own band.

**Homogeneity test.** To test whether a moment depends on time, the sample is split into $B$ non-overlapping blocks of $n$ observations, and the block estimates $\hat\theta_i$ are compared. The moment is the mean of $r_t$, of $(r_t-\bar r)^2$, or of $\hat\varepsilon_t\hat\varepsilon_{t-1}$:

$$Q = \frac{n\sum_{i=1}^{B}(\hat\theta_i-\bar\theta)^2}{\hat\sigma^2_{\text{within}}} \;\overset{H_0}{\sim}\; \chi^2(B-1), \qquad \hat\sigma^2_{\text{within}} = \frac{1}{nB-B}\sum_{i=1}^{B}\sum_{t\in i}\big(v_t-\hat\theta_i\big)^2 .$$

$Q/(B-1)$ is exactly the global F-statistic of a regression of the series on block dummies.

**Table 2.** Homogeneity test. H₀: the same moment in all blocks.

| | Daily, 230 blocks of 50 days | | | Monthly, 45 blocks of 12 months | | |
|---|---:|---:|---:|---:|---:|---:|
| | Mean | Variance | Autocov. | Mean | Variance | Autocov. |
| Q | 201.7 | 2,017.2 | 243.1 | 42.5 | 108.8 | 40.0 |
| Q / df | 0.88 | 8.81 | 1.06 | 0.97 | 2.47 | 0.91 |
| 5% critical value | 265.3 | 265.3 | 265.3 | 60.5 | 60.5 | 60.5 |
| p-value | 0.903 | 0.000 | 0.249 | 0.535 | 0.000 | 0.646 |

**Results.**
- **Mean.** Equality across blocks is not rejected at either frequency. A linear trend fitted on the 230 block means has a slope of 0.0003% per year (t = 0.18).
- **Autocovariance.** Equality is not rejected either.
- **Variance.** Equality is strongly rejected (Q/df = 8.8), and the standard deviation ranges from 1.69% to 2.60% across four equal sub-periods. This does not contradict stationarity. The block test picks up the clustering of the *conditional* variance (Fact 6), which is compatible with a constant *unconditional* variance. The block variances show no drift: they return to the same level after each episode (1987, 1990–91, 1998–2002, 2008–09, 2020). That is the behaviour of a stationary but persistent conditional variance, not of a permanent change.

> **Conclusion.** Stationarity holds for the mean and the autocovariance. For the variance, the evidence is compatible with it.

### 3.3 Fact 2: Negative skewness

![Skewness by horizon](figures/skewness_by_horizon.png)

*Figure 3. Skewness of h-day log-returns for h = 1 to 252 trading days, with and without the October 1987 crash, and the 5% critical value of the one-sided test.*

**Method.** The skewness is computed at every horizon h from 1 to 252 trading days, not just at four calendar frequencies. For a given h, the sample can be cut into non-overlapping h-day returns in h different ways, one per starting day. The h estimates are averaged so that the result does not depend on an arbitrary starting date. The test is H₀: S = 0 against H₁: S < 0, with $z = \hat S/\sqrt{6/N}$ and $N = \lfloor 11{,}542/h \rfloor$.

**Table 3.** Skewness by horizon (trading days). Reject H₀ if z < −1.645.

| h | N | Ŝ, with crash | z | Ŝ, without crash | z |
|---:|---:|---:|---:|---:|---:|
| 1 | 11,542 | −0.106 | −4.63 | +0.166 | +7.26 |
| 5 | 2,308 | −0.344 | −6.75 | −0.218 | −4.28 |
| 21 | 549 | −0.534 | −5.11 | −0.484 | −4.63 |
| 252 | 45 | −0.453 | −1.24 | −0.395 | −1.08 |

**Results.**
- **Full sample.** Skewness is negative at all 252 horizons and significant at 133 of them, essentially every horizon up to six months.
- **Daily frequency.** The daily result is not robust: five observations out of 11,542 produce it.
  - Without the crash week, daily skewness becomes positive (+0.166).
  - It is positive or close to zero in three of the four sub-periods.
  - Daily quantiles are symmetric: −3.15% / +3.24% at 5% / 95%, and −0.94% / +1.03% at the quartiles.
- **Intermediate horizons.** The fact is robust: skewness stays negative at 5 and 21 days with and without the crash, and remains significant at 114 horizons without it.
- **One year.** The estimate is negative, but not significant with 45 observations.

$\sqrt{6/N}$ is the standard error under normality, which Fact 3 rejects. Under heavy tails the true dispersion of $\hat S$ is larger, so the z-statistics overstate significance.

> **Conclusion.** Negative skewness holds from the weekly to the monthly horizon. At the daily frequency, the negative sign comes entirely from October 1987.

### 3.4 Fact 3: Heavy tails

![QQ-plots against Student-t](figures/qq_student_t.png)

*Figure 4. QQ-plots of standardized daily log-returns against Student-t distributions with ν = 10, 5 and 3.*

**Kurtosis.** Daily kurtosis is 18.3, which gives z = 335 against a critical value of 1.645. The result does not depend on the 1987 crash (14.8 without it) and holds in every sub-period (28.2, 7.4, 18.9 and 16.1). The extreme daily returns, −32.5% and +22.4%, lie 14.7 and 10.1 standard deviations from the mean. Under a normal distribution, such events have a probability below 10⁻²³.

**How heavy?** Inverting the kurtosis of a Student-t, $K = 3 + 6/(\nu-4)$, gives ν̂ = 4.4. That estimator cannot fall below 4 by construction, because the fourth moment is infinite there. Two estimators that do not rely on the fourth moment are added:
- **Hill estimator.** It uses the $k$ largest observations of one tail and estimates the exponent $a$ in $P(X>x)\sim x^{-a}$:

$$\hat a = \Big(\frac{1}{k}\sum_{i=1}^{k}\ln\frac{X_{(i)}}{X_{(k+1)}}\Big)^{-1}, \qquad \text{s.e.}(\hat a) \approx \hat a/\sqrt{k}.$$

- **Student-t maximum likelihood**, fitted on all daily returns. For a Student-t, the tail index equals ν.

**Table 4.** Tail index of daily log-returns.

| | k = 115 (1%) | k = 288 (2.5%) | k = 577 (5%) |
|---|---:|---:|---:|
| Hill, left tail | 2.70 | 2.95 | 2.63 |
| Hill, right tail | 2.97 | 2.73 | 2.47 |
| Student-t ν, maximum likelihood | 2.82 | | |
| Student-t ν, from kurtosis | 4.39 | | |

**Results.** Tails are much heavier than the kurtosis suggests. The Hill estimates lie between 2.5 and 3.0, and the maximum-likelihood ν̂ is 2.82, consistent with the QQ-plots, where ν = 3 fits best. Three consequences follow:
1. **Infinite theoretical kurtosis.** With a tail index below 4, the theoretical kurtosis is infinite. This explains why the sample kurtosis is so unstable across sub-periods (7.4 to 28.2).
2. **Finite variance.** With a tail index above 2, the variance is finite, as second-order stationarity requires.
3. **Symmetric tails.** Both tails are equally heavy: 2.70 vs 2.97 at k = 115, a gap within one standard error of about 0.26. This confirms that daily returns show no robust asymmetry.

![Kurtosis by horizon](figures/kurtosis_by_horizon.png)

*Figure 5. Kurtosis of h-day log-returns as a function of h, and the 5% critical value of the one-sided test H₁: K > 3.*

**Table 5.** Kurtosis by horizon. The critical kurtosis is $3 + 1.645\sqrt{24/N}$; ν̂ is the Student-t with the same kurtosis.

| h | N | Critical K | K̂ | z | ν̂ | K̂, ex-crash | ν̂, ex-crash |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11,542 | 3.08 | 18.29 | 335.3 | 4.39 | 14.76 | 4.51 |
| 5 | 2,308 | 3.17 | 10.14 | 70.0 | 4.84 | 9.47 | 4.93 |
| 21 | 549 | 3.34 | 6.61 | 17.2 | 5.66 | 6.56 | 5.68 |
| 252 | 45 | 4.20 | 3.29 | 0.40 | 24.7 | 3.32 | 22.7 |

Kurtosis declines smoothly with the horizon and stays significant up to h = 161 days. Beyond that, non-rejection reflects a loss of power rather than normality: with 45 annual returns, only a kurtosis above 4.20 can be detected.

> **Conclusion.** Heavy tails hold at the daily, weekly and monthly frequencies, independently of the 1987 crash, with a tail index close to 3.

### 3.5 Fact 4: Aggregational Gaussianity

![Normal QQ-plots by frequency](figures/qq_normal_by_frequency.png)

*Figure 6. Normal QQ-plots of daily, weekly, monthly and annual log-returns.*

**Table 6.** Normality tests by frequency. H₀: returns are normal. $\text{JB}/N = \tfrac{1}{6}\big(\hat S^2 + (\hat K-3)^2/4\big)$.

| | Daily | Weekly | Monthly | Annual |
|---|---:|---:|---:|---:|
| Jarque-Bera | 112,450 | 7,739 | 199.3 | 13.75 |
| Jarque-Bera p-value | 0.000 | 0.000 | 0.000 | 0.001 |
| JB / N | 9.74 | 3.24 | 0.36 | 0.31 |
| Lilliefors p-value | < 0.001 | < 0.001 | < 0.001 | 0.090 |
| Anderson-Darling (5% c.v.) | 212.9 (0.752) | 24.2 (0.752) | 4.12 (0.751) | 0.66 (0.739) |
| Kurtosis | 18.29 | 11.81 | 5.55 | 4.90 |

**Results.** The QQ-plots straighten as the horizon grows, and kurtosis falls monotonically from 18.3 to 4.9. The Jarque-Bera statistic is proportional to N, which drops from 11,542 to 45, so its decline partly reflects sample size. JB/N is a pure shape measure that can be compared across frequencies: it is divided by 27 between the daily and the monthly frequency, then stabilises.

Convergence is incomplete. At one year, Jarque-Bera still rejects normality (p = 0.001), while Lilliefors and Anderson-Darling do not. The disagreement is a matter of power. Jarque-Bera targets skewness and kurtosis, exactly where annual returns deviate (Ŝ = −0.96, K̂ = 4.90). The other two tests compare the whole distribution function using only 45 points.

> **Conclusion.** Returns move towards normality when aggregated, mostly by the monthly horizon, but normality is not reached at one year.

### 3.6 Fact 5: No linear autocorrelation

![ACF of returns](figures/acf_returns.png)

*Figure 7. Autocorrelation function of daily, weekly and monthly log-returns (40 lags) with Bartlett bands $\pm1.96/\sqrt{N}$.*

**Table 7.** Left: Ljung-Box test, H₀: ρ₁ = … = ρₘ = 0. Right: regression $r_t = c + \sum_{j=1}^{5}\phi_j r_{t-j} + u_t$ and F-test of H₀: φ₁ = … = φ₅ = 0 under OLS and White covariance matrices.

| | ρ̂₁ | m | Q_LB(m) | p-value | φ̂₁ | t OLS | t White | F OLS (p) | F White (p) | R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Daily | −0.032 | 25 | 62.6 | 0.000 | −0.033 | −3.53 | −1.53 | 5.08 (0.000) | 0.98 (0.426) | 0.2% |
| Weekly | −0.081 | 25 | 94.8 | 0.000 | −0.077 | −3.76 | −1.78 | 4.03 (0.001) | 0.93 (0.460) | 0.8% |
| Monthly | 0.017 | 25 | 32.1 | 0.154 | 0.028 | 0.64 | 0.47 | 3.51 (0.004) | 2.62 (0.024) | 3.2% |
| Annual | −0.106 | 10 | 20.1 | 0.028 | — | — | — | — | — | — |

**Results.**
- **Ljung-Box.** Autocorrelations are tiny, yet the Ljung-Box test rejects at the daily and weekly frequencies. The test relies on $\text{Var}(\hat\rho_j) = 1/N$, which understates the true dispersion when volatility is clustered.
- **White standard errors.** The first-lag t-statistic falls from −3.53 to −1.53 (daily) and from −3.76 to −1.78 (weekly), and the joint test no longer rejects. The Ljung-Box rejections are produced by heteroskedasticity. In any case, the predictable share of the variance is negligible (R² of 0.2% and 0.8%).
- **Monthly lag 4.** At the monthly frequency, one lag survives: φ̂₄ = −0.171 (White t = −3.57). It is isolated, with no counterpart at neighbouring lags and no obvious economic interpretation. It is reported as a deviation, not as exploitable predictability.
- **Annual.** The annual Ljung-Box rejection rests on 45 observations; the Box-Pierce version does not reject (p = 0.065).

> **Conclusion.** Returns are not autocorrelated at the daily and weekly frequencies once heteroskedasticity is accounted for. One isolated negative autocorrelation remains at the monthly lag 4.

### 3.7 Fact 6: Volatility clustering and long memory

![ACF of squared returns](figures/acf_squared_returns.png)

*Figure 8. Autocorrelation function of squared daily, weekly and monthly log-returns (40 lags).*

**Table 8.** Autocorrelation of squared log-returns, by frequency and, for daily returns, by sub-period.

| Lag | Daily | Weekly | Monthly | | Lag | 1980–91 | 1991–03 | 2003–14 | 2014–25 |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.258 | 0.378 | 0.195 | | 1 | 0.048 | 0.245 | 0.353 | 0.427 |
| 5 | 0.178 | 0.063 | 0.080 | | 5 | 0.074 | 0.125 | 0.227 | 0.290 |
| 10 | 0.122 | 0.079 | 0.015 | | 20 | 0.010 | 0.057 | 0.138 | 0.084 |
| 20 | 0.091 | 0.038 | 0.026 | | 60 | 0.005 | 0.034 | 0.091 | 0.037 |
| Band | 0.018 | 0.040 | 0.084 | | Band | 0.037 | 0.037 | 0.037 | 0.037 |

**Results.** Large returns of either sign occur in clusters: 1987, 1990–91, 1998–2002, 2008–09 and 2020.
- **Size of the effect.** The first 25 autocorrelations of squared daily returns are all positive, between 0.088 and 0.258, i.e. 5 to 14 times the band. The Ljung-Box statistic reaches 5,939 at lag 25, and Engle's ARCH-LM test ($NR^2\sim\chi^2(m)$) gives 770 with one lag and 1,581 with 25 lags. Unlike Fact 5, the effect is so large that the calibration of the test does not matter.
- **By frequency.** Persistence weakens as the sampling interval grows. Daily autocorrelations are still five times the band at lag 20, weekly ones fall inside it from about lag 20, and monthly ones are significant at lag 1 only.
- **By sub-period.** Clustering is present in all four sub-periods. It is weak in 1980–91 only, where the single observation of 19 October 1987 dominates the variance of $r_t^2$.

**GARCH(1,1).** The conditional variance is modelled as

$$h_t = \omega + \alpha\,(r_{t-1}-\bar r)^2 + \beta\,h_{t-1},$$

estimated by Gaussian maximum likelihood, with the likelihood coded directly. $\alpha$ measures how tomorrow's variance reacts to today's shock, and $\alpha+\beta$ how persistent that reaction is. If the model captures the clustering, the standardized residuals $z_t = (r_t-\bar r)/\sqrt{h_t}$ should show none.

**Table 9.** GARCH(1,1) on daily log-returns in percent.

| Parameter | Estimate | | Diagnostic | Value |
|---|---:|---|---|---:|
| ω (%²) | 0.046 | | Ljung-Box(25) on $r_t^2$ | 5,939.2 (p = 0.000) |
| α | 0.083 | | Ljung-Box(25) on $z_t^2$ | 25.7 (p = 0.426) |
| β | 0.909 | | Kurtosis of $z_t$ | 10.19 |
| α + β | 0.993 | | Implied unconditional st. dev. | 2.48% (sample: 2.20%) |
| Half-life of a shock | 92.8 days | | | |

![GARCH conditional volatility](figures/garch_volatility.png)

*Figure 9. Daily log-returns and GARCH(1,1) conditional volatility.*

**GARCH results.**
- **Persistence.** α + β = 0.993: a volatility shock loses half its effect only after 93 trading days, more than four months. Persistence is below one, so the model is covariance-stationary with a finite unconditional variance. Clustering and a constant unconditional variance are therefore compatible, as argued for Fact 1.
- **Fit.** After filtering, the Ljung-Box statistic on squared residuals falls from 5,939 to 25.7 (p = 0.43): a single GARCH(1,1) captures all the autocorrelation of squared returns.
- **Residual kurtosis.** The kurtosis of $z_t$ (10.2) is lower than that of the returns (18.3) but far above 3. Clustering explains part of the heavy tails, not all of them: the conditional distribution is itself heavy-tailed.

> **Conclusion.** Volatility clustering holds strongly, with persistence that decreases with the sampling interval, and a GARCH(1,1) with persistence 0.993 describes it fully.

### 3.8 Fact 7: Leverage effect

![Cross-correlation between past returns and squared returns](figures/leverage_cross_correlation.png)

*Figure 10. Cross-correlation $\text{corr}(r_{t-j}, r_t^2)$, daily returns 1988–2025, with bands $\pm1.96/\sqrt{N}$.*

**Standard test.** On 1988–2025, the cross-correlations between past returns and current squared returns are negative at 9 of the first 10 lags: −0.084, −0.068 and −0.057 at j = 1, 2, 3, against a band of ±0.020. The joint statistic $Q = N\sum_{j=1}^{10}\hat\rho_j^2 = 206.5$ far exceeds the χ²(10) critical value of 18.3. However, the same rule also flags several correlations at *negative* lags as significant (j = −1: z = 5.5; j = −9: z = 5.8). No leverage effect can exist there, which shows the $1/N$ variance is too small under volatility clustering.

**Robust test.** Squared returns are regressed on ten lagged returns with White standard errors. A leverage effect implies γⱼ < 0.

**Table 10.** $r_t^2 = c + \sum_{j=1}^{10}\gamma_j r_{t-j} + u_t$, daily returns (first three lags shown).

| | 1988–2025 | | | Full sample (with 1987) | | |
|---|---:|---:|---:|---:|---:|---:|
| | γ̂ⱼ × 10³ | t OLS | t White | γ̂ⱼ × 10³ | t OLS | t White |
| $r_{t-1}$ | −7.46 | −8.70 | −2.12 | −7.50 | −8.85 | −2.35 |
| $r_{t-2}$ | −6.61 | −7.71 | −2.39 | −6.08 | −7.17 | −2.50 |
| $r_{t-3}$ | −5.26 | −6.13 | −2.00 | −4.97 | −5.86 | −2.14 |
| F-test, all 10 lags (p) | | 23.46 (0.000) | 1.63 (0.091) | | 21.78 (0.000) | 1.86 (0.047) |
| R² | 2.4% | | | 1.9% | | |

**Results for JPM.** Nine of the ten coefficients are negative in both samples, and the first three remain significant with White standard errors. The joint evidence is much weaker than Q suggests: the robust F-test rejects at 5% on the full sample (p = 0.047), and only at 10% on 1988–2025 (p = 0.091). Economically, a −2% daily return raises the next day's expected squared return by about 1.5 × 10⁻⁴, i.e. 28% of its average level.

![VIX changes versus JPM returns](figures/vix_vs_jpm_returns.png)

*Figure 11. Daily change of the VIX against the JPM daily log-return, 1990–2025.*

**Table 11.** Regressions of the daily change of the VIX on the JPM daily log-return (1990–2025, N = 9,066). The asymmetric model is $\Delta VIX_t = a + \beta^-\min(r_t,0) + \beta^+\max(r_t,0) + e_t$.

| | Estimate | t OLS | t White |
|---|---:|---:|---:|
| β (symmetric model) | −38.0 | −58.2 | −24.9 |
| β⁻ (negative returns) | −49.0 | −44.5 | −16.3 |
| β⁺ (positive returns) | −27.9 | −26.8 | −12.8 |
| F-test β⁻ = β⁺ | | 152.0 | 24.0 |

**Results at market level.** The leverage effect is clear here. The VIX change is negatively related to the JPM return: the correlation is −0.52, and a −1% JPM return is associated with a 0.38-point rise in the VIX. The response is also asymmetric: a fall in JPM raises implied volatility almost twice as much as an equal rise lowers it. Both results survive White standard errors. The VIX is the implied volatility of the S&P 500, so this regression documents the leverage effect of the market, of which JPM is a high-beta component, rather than that of JPM alone.

> **Conclusion.** The leverage effect holds in sign and at the first lags. Its joint significance in JPM's own returns is borderline once heteroskedasticity is accounted for.

---

## 4. Part II: Factor models

### 4.1 Specification

Monthly JPM excess returns, in percent and net of the one-month T-bill rate, are regressed on the Fama-French factors. MKT is the excess return of the CRSP value-weighted index, SMB is small minus big, and HML is high minus low book-to-market. The sample runs from April 1980 to December 2025 (T = 549).

$$\begin{aligned}
\text{Model 0:}\quad & R_t - R_{f,t} = \alpha + \varepsilon_t \\
\text{CAPM:}\quad & R_t - R_{f,t} = \alpha + \beta_{M}\,MKT_t + \varepsilon_t \\
\text{Fama-French:}\quad & R_t - R_{f,t} = \alpha + \beta_{M}\,MKT_t + \beta_{S}\,SMB_t + \beta_{H}\,HML_t + \varepsilon_t
\end{aligned}$$

### 4.2 Estimates

**Table 12.** OLS estimates, JPM monthly excess simple returns. OLS standard errors in parentheses, White (HC0) in brackets. Significance with OLS standard errors: \* 10%, \*\* 5%, \*\*\* 1%.

| | Model 0 | CAPM | Fama-French |
|---|:---:|:---:|:---:|
| α (% per month) | 1.124\*\*\* | 0.192 | −0.086 |
| | (0.381) [0.381] | (0.304) [0.310] | (0.284) [0.289] |
| MKT | | 1.237\*\*\* | 1.345\*\*\* |
| | | (0.067) [0.078] | (0.065) [0.075] |
| SMB | | | 0.158 |
| | | | (0.098) [0.114] |
| HML | | | 0.871\*\*\* |
| | | | (0.092) [0.104] |
| R² | 0.000 | 0.382 | 0.470 |
| Adjusted R² | 0.000 | 0.381 | 0.467 |
| Global F-statistic | — | 338.4 | 161.2 |
| Residual Jarque-Bera | 63.5 | 184.5 | 169.8 |
| Residual Durbin-Watson | 1.98 | 2.11 | 2.10 |

![CAPM fitted line](figures/capm_fitted_line.png)

*Figure 12. JPM monthly excess return against the CRSP value-weighted excess return, with the fitted market-model line ŷ = 0.192 + 1.237x.*

**Inference.** Residuals are far from normal (Jarque-Bera between 63 and 185, against a critical value of 5.99), in line with Facts 3 and 4. White standard errors of the slopes are 14% to 16% larger than the OLS ones, a sign of heteroskedasticity. With T = 549 the tests remain asymptotically valid; conclusions are based on both covariance matrices. Durbin-Watson statistics close to 2 show no first-order autocorrelation of the residuals.

### 4.3 Tests of restrictions

**Table 13.** F-tests of linear restrictions under the OLS and White covariance matrices. For a single restriction, F = t².

| Hypothesis | F (OLS) | p-value | F (White) | p-value |
|---|---:|---:|---:|---:|
| CAPM: α = 0 | 0.40 | 0.528 | 0.38 | 0.536 |
| CAPM: β_M = 1 | 12.46 | 0.000 | 9.29 | 0.002 |
| Fama-French: α = 0 | 0.09 | 0.762 | 0.09 | 0.766 |
| Fama-French: β_M = 1 | 28.13 | 0.000 | 21.13 | 0.000 |
| Fama-French: β_S = β_H = 0 | 45.20 | 0.000 | 35.76 | 0.000 |

### 4.4 Interpretation

**Market beta.** In the CAPM, β̂ = 1.24 (t = 18.4). A 1% market excess return is associated with a 1.24% JPM excess return. In a month when the market loses 5% in excess of the risk-free rate, the model predicts −6.0% for JPM. For a single stock, the informative null is β = 1 rather than β = 0. With t = (1.237 − 1)/0.067 = 3.53, it is rejected, also with White standard errors (p = 0.002). JPM is significantly riskier than the market, as expected from a highly levered bank with strongly pro-cyclical earnings. The beta rises to 1.35 once SMB and HML are included.

**Value.** The HML loading is large and highly significant: 0.87, with t = 9.5 (8.4 with White standard errors). Holding the market fixed, a 1% return on the value-minus-growth portfolio is associated with a 0.87% JPM return. JPM behaves like a value stock, which is typical of banks and their high book-to-market ratios.

**Size.** The SMB loading is small (0.16) and not significant, even at 10% (t = 1.62; 1.39 with White standard errors). As one of the largest US companies, JPM was not expected to load positively on the size factor.

**Joint test.** SMB and HML are jointly highly significant, entirely because of HML:

$$F = \frac{(R^2_{FF} - R^2_{CAPM})/2}{(1 - R^2_{FF})/(T-4)} = \frac{(0.4701 - 0.3822)/2}{0.5299/545} = 45.2,$$

against a 5% critical value of 3.0 (35.8 with White standard errors).

**Goodness of fit.** The market factor explains 38.2% of the variance of JPM excess returns. Adding SMB and HML raises the adjusted R² from 0.381 to 0.467. About 53% of the variance remains idiosyncratic, which is usual for a single stock. In theory, that risk can be diversified away and earns no premium.

**Alphas.**
- **Model 0.** The constant is JPM's average excess return: 1.12% per month, about 13.5% per year, significant at 1% (t = 2.95).
- **CAPM.** The CAPM implies $E(R_i - R_f) = \beta_i\,E(R_M - R_f)$, hence α = 0. The estimate is +0.19% per month and far from significant (t = 0.63), so the data are compatible with the CAPM. Because the regression line passes through the sample means, the average excess return decomposes as 1.124 = α̂ + β̂·x̄ = 0.192 + 1.237 × 0.753. The market premium over the sample was 0.75% per month, JPM's beta entitles it to 0.93%, and it earned an insignificant 0.19% more.
- **Fama-French.** In a multi-factor model of the APT type, absence of arbitrage implies $E(R_i - R_f) = \sum_k \beta_{ik}\lambda_k$. When the factors are themselves excess returns or long-short portfolios, the intercept of the time-series regression should again be zero. The estimate is −0.09% per month (t = −0.30): JPM earned exactly what its market and value exposures imply.

### 4.5 Simple versus log-returns

The Fama-French factors are simple returns. The regressions therefore use simple returns for JPM, $R_t = P_t/P_{t-1} - 1$, so that $R_t - R_f$ is the return on a self-financing position, the quantity that the CAPM and the APT price. Part I uses log-returns, so the models are re-estimated with $\ln(P_t/P_{t-1})$ for comparison. Month by month the difference is of second order, $\ln(1+R_t) \approx R_t - R_t^2/2$, but on average it is not. The mean log-return is lower by about $\sigma^2/2 = 9.04^2/200 \approx 0.41\%$ per month, the same order of magnitude as the intercepts.

**Table 14.** Estimates with JPM simple returns and log-returns (OLS t-statistics in parentheses).

| | Simple returns | Log-returns |
|---|---:|---:|
| Model 0: α | 1.124 (2.95) | 0.713 (1.85) |
| CAPM: α | 0.192 (0.63) | −0.244 (−0.80) |
| CAPM: MKT | 1.237 (18.40) | 1.272 (18.84) |
| CAPM: R² | 0.382 | 0.394 |
| Fama-French: α | −0.086 (−0.30) | −0.521 (−1.83) |
| Fama-French: MKT | 1.345 (20.66) | 1.378 (21.08) |
| Fama-French: SMB | 0.158 (1.62) | 0.172 (1.75) |
| Fama-French: HML | 0.871 (9.51) | 0.872 (9.48) |
| Fama-French: R² | 0.470 | 0.479 |

All three intercepts shift by 0.41 to 0.44 percentage points, exactly the predicted amount, while slopes and R² are almost unchanged. With log-returns, the Fama-French alpha looks negative and significant at 10% (t = −1.83). That result is an artefact of mixing the two return conventions.

> **Conclusion.** Without factors, the intercept is a positive risk premium, significant at 1%. Once the market factor, or the three factors, are included, the intercept is statistically indistinguishable from zero. The results are compatible with both the CAPM and the multi-factor view. The three-factor model gives the better description of JPM's risk: a beta well above 1 and a strong value tilt.

---

## 5. Limitations

- **Single asset.** All conclusions concern one stock. Before 2000, the price history is that of JPMorgan Chase's predecessor entities, and the business changed through several major mergers.
- **Normal-based tests.** Skewness and kurtosis z-tests use standard errors that are only valid under normality. Under heavy tails they overstate significance, which is why robust complements are reported alongside them.
- **GARCH specification.** The GARCH(1,1) uses a Gaussian likelihood, but standardized residuals keep a kurtosis of 10.2. A Student-t or asymmetric (GJR, EGARCH) specification would be a natural extension, the latter also modelling the leverage effect directly.
- **Market-level VIX.** The VIX measures S&P 500 implied volatility, so the VIX regressions capture the market's leverage effect rather than JPM's.
- **Constant betas.** Factor loadings are assumed constant over 45 years. Rolling or regime-dependent betas, and additional factors (momentum, profitability, investment), are left for further work.
- **Data revisions.** Yahoo Finance may revise historical prices, so re-running the notebook can produce slightly different figures.

---

## 6. Reproducing the results

```bash
pip install -r requirements.txt
jupyter notebook JPM_financial_econometrics.ipynb
```

- **Python.** Requires Python 3.10 or later (developed with Python 3.14).
- **Downloads.** JPM and VIX prices are downloaded at runtime through `yfinance`, so an internet connection is needed.
- **Fama-French file.** `FF_Research_Data_Factors_monthly.csv` must be in the same folder as the notebook. It contains the monthly block of the *Fama/French 3 Factors* file from the Kenneth R. French Data Library.

### Repository structure

```
├── JPM_financial_econometrics.ipynb    # full analysis: code, tables and figures
├── FF_Research_Data_Factors_monthly.csv
├── figures/                            # figures used in this README
├── requirements.txt
└── README.md
```

### Main libraries

`pandas`, `numpy`, `scipy` (distributions, tests, optimisation), `statsmodels` (OLS, robust covariance, diagnostic tests), `matplotlib`, `yfinance`.
