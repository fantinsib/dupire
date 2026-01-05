##### SVI model 
# 
# k = log(K/Ft) -> log moneyness
# a -> verical shift
# b > 0 -> smile amplitude 
# p in (-1,1) -> skew 
# m -> horizontal translation
# sigma > 0 -> lissage
#

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import least_squares

@dataclass(frozen=True)
class SVIParams:
    """
    Args:
        log_moneyness: log moneyness (log(K/F))
        a: minimum total variance
        b: slope. must be stricly positive
        rho: skew. must be in (-1,1)
        m: center, location of smile minimum
        sigma: curvature; must be strictly positive
    """
    a: float
    b : float
    rho: float 
    m: float 
    sigma: float 

class SVI:
    
    def compute(self, log_moneyness: np.array, params: SVIParams):
        """
        Args:
        params: SVIParams object

        Returns:
            w(k): total implied variance

        """
        p2 = np.sqrt((log_moneyness-params.m)**2 + params.sigma**2)
        p1 = params.rho*(log_moneyness-params.m)
        return params.a + params.b*(p1+p2)


    def fit(self, log_moneyness, w, weights=None, init=None, max_nfev=10000):
        """
        Returns the parameters of SVI with the best fit from the data

        Args: 


        """
        log_moneyness = np.asarray(log_moneyness, dtype=float).ravel()
        w = np.asarray(w, dtype=float).ravel()

        if log_moneyness.shape != w.shape:
            raise ValueError(f"log_moneyness and w must have same shape, got {log_moneyness.shape} vs {w.shape}")
        if log_moneyness.size < 5:
            raise ValueError("Need at least 5 points to fit 5 SVI parameters robustly.")
        if np.any(~np.isfinite(log_moneyness)) or np.any(~np.isfinite(w)):
            raise ValueError("log_moneyness and w must be finite.")
        if np.any(w < 0):
            raise ValueError("Total variance w must be non-negative.")

        if weights is not None:
            weights = np.asarray(weights, dtype=float).ravel()
            if weights.shape != log_moneyness.shape:
                raise ValueError("weights must have same shape as log_moneyness and w.")
            if np.any(weights < 0) or np.any(~np.isfinite(weights)):
                raise ValueError("weights must be finite and non-negative.")
            sqrt_wt = np.sqrt(weights)
        else:
            sqrt_wt = None


        user_init = init is not None

        if init is None:
            i_min = int(np.argmin(w))
            m0 = float(log_moneyness[i_min])

            # sigma controls "width"; use a fraction of log_moneyness-range as a starting point
            log_moneyness_range = float(np.max(log_moneyness) - np.min(log_moneyness))
            sigma0 = max(1e-3, 0.25 * log_moneyness_range)  # crude but usually Olog_moneyness

            # b controls slope magnitude; use robust scale from w spread
            w_spread = float(np.percentile(w, 90) - np.percentile(w, 10))
            b0 = max(1e-6, w_spread / (log_moneyness_range + 1e-6))

            # rho: skew; equity often negative, but keep neutral-ish initial guess
            rho0 = -0.2

            # a: set so that w(m) approximately matches
            # at log_moneyness=m: w ≈ a + b*sigma
            a0 = max(0.0, float(w[i_min]) - b0 * sigma0)

            init = SVIParams(a=a0, b=b0, rho=rho0, m=m0, sigma=sigma0)

        # Build a few fallback inits to improve convergence on tough slices.
        if user_init:
            inits = [init]
        else:
            inits = []
            rho_guesses = [-0.7, -0.2, 0.2]
            sigma_guesses = [init.sigma * s for s in (0.5, 1.0, 2.0)]
            b_guesses = [init.b]
            for rho0 in rho_guesses:
                for sigma0 in sigma_guesses:
                    for b0 in b_guesses:
                        a0 = max(0.0, float(np.min(w)) - b0 * sigma0)
                        inits.append(SVIParams(a=a0, b=b0, rho=rho0, m=init.m, sigma=max(1e-8, sigma0)))

        # ---------- Bounds ----------
        # a can be slightly negative in raw fits, but for stability keep it >= -eps.
        # If you want stricter positivity, set a_low = 0.0
        a_low = -1e-6
        a_high = np.inf
        b_low, b_high = 1e-12, np.inf
        rho_low, rho_high = -0.999, 0.999
        m_low, m_high = float(np.min(log_moneyness) - 2.0), float(np.max(log_moneyness) + 2.0)
        sig_low, sig_high = 1e-8, np.inf

        lb = np.array([a_low, b_low, rho_low, m_low, sig_low], dtype=float)
        ub = np.array([a_high, b_high, rho_high, m_high, sig_high], dtype=float)

        # ---------- Residuals ----------
        def residuals(x: np.ndarray) -> np.ndarray:
            a, b, rho, m, sigma = x
            params = SVIParams(a,b, rho, m, sigma)
            w_model = self.compute(log_moneyness, params)
            r = (w_model - w)
            if sqrt_wt is not None:
                r = r * sqrt_wt
            return r

        best_res = None
        for cand in inits:
            x0 = np.array([cand.a, cand.b, cand.rho, cand.m, cand.sigma], dtype=float)
            res = least_squares(
                residuals,
                x0=x0,
                bounds=(lb, ub),
                method="trf",
                loss="linear",
                f_scale=1.0,
                max_nfev=max_nfev,
                    x_scale="jac",
            )
            if best_res is None or res.cost < best_res.cost:
                best_res = res
            if res.success:
                best_res = res
                break

        if best_res is None or not best_res.success:
            msg = best_res.message if best_res is not None else "no result"
            raise RuntimeError(f"SVI fit failed after {len(inits)} inits: {msg}")

        a, b, rho, m, sigma = best_res.x
        return SVIParams(float(a), float(b), float(rho), float(m), float(sigma))











def plot_svi_interactive_html(
    output_html="svi_interactive.html",
    k_min=-1.0,
    k_max=1.0,
    n_k=400,
):
    import webbrowser
    from pathlib import Path
    svi = SVI()

    k = np.linspace(k_min, k_max, n_k)
    k_js = ",".join(f"{v:.6f}" for v in k)
    # Fixed y-scale based on parameter bounds
    bounds = {
        "a": (-1.0, 1.0),
        "b": (0.0, 3.0),
        "rho": (-0.99, 0.99),
        "m": (-1.0, 1.0),
        "sigma": (0.01, 2.0),
    }
    y_min = float("inf")
    y_max = float("-inf")
    for a in bounds["a"]:
        for b in bounds["b"]:
            for rho in bounds["rho"]:
                for m in bounds["m"]:
                    for sigma in bounds["sigma"]:
                        params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
                        w = svi.compute(k, params)
                        y_min = min(y_min, float(np.min(w)))
                        y_max = max(y_max, float(np.max(w)))
    y_min -= 0.1 * abs(y_min)
    y_max += 0.1 * abs(y_max)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SVI Interactive</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: "Fira Sans", Arial, sans-serif;
      background: #0f141b;
      color: #e6e6e6;
    }}
    .container {{
      max-width: 980px;
      margin: 24px auto;
      padding: 0 16px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px 24px;
      margin-bottom: 16px;
    }}
    .control label {{
      display: block;
      font-size: 14px;
      margin-bottom: 4px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    #plot {{
      background: #0f141b;
      border: 1px solid #2b3340;
      border-radius: 8px;
    }}
    @media (max-width: 720px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h2>SVI total variance</h2>
    <div class="controls">
      <div class="control">
        <label>a: <span id="a_val">0.20</span></label>
        <input id="a" type="range" min="-1.0" max="1.0" step="0.01" value="0.20" />
      </div>
      <div class="control">
        <label>b: <span id="b_val">0.50</span></label>
        <input id="b" type="range" min="0.0" max="3.0" step="0.01" value="0.50" />
      </div>
      <div class="control">
        <label>rho: <span id="rho_val">0.00</span></label>
        <input id="rho" type="range" min="-0.99" max="0.99" step="0.01" value="0.00" />
      </div>
      <div class="control">
        <label>m: <span id="m_val">0.00</span></label>
        <input id="m" type="range" min="-1.0" max="1.0" step="0.01" value="0.00" />
      </div>
      <div class="control">
        <label>sigma: <span id="sigma_val">0.20</span></label>
        <input id="sigma" type="range" min="0.01" max="2.0" step="0.01" value="0.20" />
      </div>
    </div>
    <div id="plot"></div>
  </div>

  <script>
    const k = [{k_js}];
    const layout = {{
      paper_bgcolor: "#0f141b",
      plot_bgcolor: "#0f141b",
      font: {{ color: "#e6e6e6" }},
      xaxis: {{
        title: "log-moneyness k",
        gridcolor: "#2b3340",
        zerolinecolor: "#2b3340"
      }},
      yaxis: {{
        title: "w(k)",
        gridcolor: "#2b3340",
        zerolinecolor: "#2b3340",
        range: [-0.2, 2]
      }},
      margin: {{ l: 50, r: 20, t: 20, b: 45 }}
    }};

    function sviCurve(a, b, rho, m, sigma) {{
      const y = new Array(k.length);
      for (let i = 0; i < k.length; i++) {{
        const km = k[i] - m;
        const p2 = Math.sqrt(km * km + sigma * sigma);
        const p1 = rho * km;
        y[i] = a + b * (p1 + p2);
      }}
      return y;
    }}

    function update() {{
      const a = parseFloat(document.getElementById("a").value);
      const b = parseFloat(document.getElementById("b").value);
      const rho = parseFloat(document.getElementById("rho").value);
      const m = parseFloat(document.getElementById("m").value);
      const sigma = parseFloat(document.getElementById("sigma").value);

      document.getElementById("a_val").textContent = a.toFixed(2);
      document.getElementById("b_val").textContent = b.toFixed(2);
      document.getElementById("rho_val").textContent = rho.toFixed(2);
      document.getElementById("m_val").textContent = m.toFixed(2);
      document.getElementById("sigma_val").textContent = sigma.toFixed(2);

      const y = sviCurve(a, b, rho, m, sigma);
      const trace = {{
        x: k,
        y: y,
        mode: "lines",
        line: {{ color: "#00d0ff", width: 2 }}
      }};
      Plotly.react("plot", [trace], layout, {{ responsive: true }});
    }}

    ["a", "b", "rho", "m", "sigma"].forEach((id) => {{
      document.getElementById(id).addEventListener("input", update);
    }});

    update();
  </script>
</body>
</html>
"""

    output_path = Path(output_html).resolve()
    output_path.write_text(html, encoding="utf-8")
    webbrowser.open(output_path.as_uri())
