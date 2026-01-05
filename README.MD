# Dupire Volatility Modeling 

This project aims at modeling local volatility using volatility surface and Dupire's model. 

## Methodology 

Data is generated from the make_synth_iv_chain_svi() function (code and comments are in option_chain_generator.py). 

The option chain used in the following notes has been generated with parameters :

```python
S0=100.0, 
r=0.02,
q=0.00,
maturities_day = (30, 60, 90, 180, 270, 360),
k_min=-0.30, 
k_max=0.30, 
n_k=31,
seed=0
```

We obtain data that present a visible smile (feel free to check the Jupyter notebook for interactive 3D plots).

![alt text](img/smile.png)


## SVI Fit

From this data, an SVI curve is fitted for each maturity. SVI model gives the following equation :

$$ w(k) = a + b(\rho (k-m) + \sqrt{(k-m)^2 +\sigma^2)} $$

with :
***
$w(k)$ : the total variance 

$a$ : minimum total variance

$b$: slope of the curve ($b>0$)

$\rho$: the skew with $\rho \space \epsilon \space(-1,1)$

$m$: center, location of smile minimum

$\sigma$: curvature, with $\sigma > 0$

***

Note : the **[svi_interactive.html](svi_interactive.html)** file allows to play with the SVI parameters and visualize the resulting curve.


### Fit results

![alt text](img/svi_fit_per_mat.png)

![alt text](img/tiv_scatter_svi_fitted.png)