import numpy as np
import plotly.graph_objects as go

import option_chain_generator as oc

mat_days = np.arange(7, 256, 7)
#df = oc.make_synth_iv_chain(seed=12, maturities_days=mat_days, sigma0 = 0.5, noise_atm=0.05, noise_wings=0.05, k_min=-0.2, k_max=0.2, n_k=31)
df = oc.make_synth_iv_chain(maturities_days=mat_days)
df2 = df.copy()
df2["k"] = np.log(df2["K"] / df2["F"])
#df2 = df2[df2["k"] > 0]


df2 = df2.drop_duplicates(subset=["T", "k"]).sort_values(["T", "k"])

T_vals = np.sort(df2["T"].unique())
k_vals = np.sort(df2["k"].unique())

iv_mat = (
    df2.pivot(index="T", columns="k", values="iv_mid")
       .reindex(index=T_vals, columns=k_vals)
       .to_numpy()
)

fig = go.Figure(
    data=go.Scatter3d(
        x=df2["k"],
        y=df2["T"],
        z=df2["iv_mid"],
        mode="markers",
        marker=dict(size=6, color=df2["iv_mid"], colorscale="Turbo")
    )
)

fig.update_layout(
    title="Implied Vol Surface",
    template="plotly_dark",
    paper_bgcolor="#0f141b",
    plot_bgcolor="#0f141b",
    font=dict(color="#e6e6e6", family="Fira Sans, sans-serif"),
    scene=dict(
        xaxis_title="log-moneyness k = log(K/F)",
        yaxis_title="T (years)",
        zaxis_title="Implied vol",
        xaxis=dict(
            showbackground=True,
            backgroundcolor="#0f141b",
            gridcolor="#2b3340",
            zerolinecolor="#2b3340"
        ),
        yaxis=dict(
            showbackground=True,
            backgroundcolor="#0f141b",
            gridcolor="#2b3340",
            zerolinecolor="#2b3340"
        ),
        zaxis=dict(
            showbackground=True,
            backgroundcolor="#0f141b",
            gridcolor="#2b3340",
            zerolinecolor="#2b3340"
        ),
        aspectmode="cube"
    )
)

fig.show()
