import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import t
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
from scipy.stats import cauchy
from scipy.stats import f
import plotly.express as px

inicio = st.selectbox("Some problems and examples of statistics graphics. "
                      "Select the problem:", ("Distributions", "Bivariate Normal Distribution",
                                              "Central Limit Theorem", "Quantiles"))

if inicio == "Distributions":
    opcao = st.sidebar.selectbox("Distributions", ("Normal", "Exponential",
                                                   "Chi-squared", "T student", "Binomial", "Poisson"))

    np.random.seed(999)
    if opcao == "Normal":
        st.header("Probability Density Function")
        histog = st.sidebar.checkbox('Histogram')
        curve = st.sidebar.checkbox('Curve', value=True)
        mm = st.sidebar.checkbox('Mean/Median')
        mu = st.sidebar.number_input('Mean:', step=0.5)
        sd = st.sidebar.number_input('Standard Deviation:', min_value=0.0, value=1.0, step=0.5)
        nor1 = np.random.normal(mu, sd, size=1000)
        nor = np.linspace(norm.ppf(0.00000000001, mu, sd), norm.ppf(0.99999999999, mu, sd), 100)

        fig, ax = plt.subplots()
        ax.set_ylabel("Probability Density")
        if curve:
            ax.plot(nor, norm.pdf(nor, mu, sd), 'r-', lw=3, alpha=0.7, label='Normal')
        if histog:
            ax.hist(nor1, 40, range=(-8, 8), density=True, alpha=0.6)
        plt.xlim(-9, 9)
        plt.ylim(0, 0.8)
        if mm:
            plt.axvline(x=mu, color='y', ls="-.", label='Mean and Median')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig)

    elif opcao == "Exponential":
        st.header("Probability Density Function")
        histog = st.sidebar.checkbox('Histogram')
        curve = st.sidebar.checkbox('Curve', value=True)
        mm = st.sidebar.checkbox('Mean/Median')
        lambd = st.sidebar.number_input('Lambda:', min_value=0.1, value=2.0, step=0.25)
        beta = 1.0 / lambd
        e = np.random.exponential(scale=beta, size=1000)
        ex = np.linspace(expon.ppf(0.0000001, scale=beta), expon.ppf(0.9999999, scale=beta), 100)

        fig2, ax2 = plt.subplots()
        ax2.set_ylabel("Probability Density")
        if histog:
            ax2.hist(e, 60, range=(-9, 9), density=True, alpha=0.6)
        if curve:
            ax2.plot(ex, expon.pdf(ex, scale=beta), 'r-', lw=3, alpha=0.7, label='Exponential')
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        if mm:
            plt.axvline(x=expon.mean(loc=0, scale=beta), color='y', ls="-.", label='Mean')
            plt.axvline(x=expon.median(loc=0, scale=beta), color='g', ls="-", label='Median')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig2)

    elif opcao == "Chi-squared":
        st.header("Probability Density Function")
        histog = st.sidebar.checkbox('Histogram')
        curve = st.sidebar.checkbox('Curve', value=True)
        mm = st.sidebar.checkbox('Mean/Median')
        df = st.sidebar.number_input('Degrees of freedom:', min_value=1, value=3, step=1)
        qs1 = np.random.chisquare(df, 100)
        qs = np.linspace(chi2.ppf(0.00000001, df), chi2.ppf(0.99999999, df), 100)

        fig3, ax3 = plt.subplots()
        ax3.set_ylabel("Probability Density")
        if histog:
            ax3.hist(qs1, 10, range=(0, 40), density=True, alpha=0.6)
        if curve:
            ax3.plot(qs, chi2.pdf(qs, df), 'r-', lw=3, alpha=0.7, label='Qui-quadrado')
        plt.xlim(0, 40)
        plt.ylim(0, 0.25)
        if mm:
            plt.axvline(x=chi2.mean(df, loc=0, scale=1), color='y', ls="-.", label='Mean')
            plt.axvline(x=chi2.median(df, loc=0, scale=1), color='g', ls="-", label='Median')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig3)

    elif opcao == "T student":
        st.header("Probability Density Function")
        histog = st.sidebar.checkbox('Histogram')
        curve = st.sidebar.checkbox('Curve', value=True)
        mm = st.sidebar.checkbox('Mean/Median')
        df2 = st.sidebar.number_input('Degrees of Freedom:', min_value=1, value=4, step=1)
        ts1 = np.random.standard_t(df2, size=100)
        ts = np.linspace(t.ppf(0.000001, df2), t.ppf(0.999999, df2), 100000)

        fig4, ax4 = plt.subplots()
        ax4.set_ylabel("Probability Density")
        if histog:
            ax4.hist(ts1, 40, range=(-9, 9), density=True, alpha=0.6)
        if curve:
            ax4.plot(ts, t.pdf(ts, df2), 'r-', lw=3, alpha=0.7, label='T Student')
        plt.xlim(-9, 9)
        plt.ylim(0, 0.4)
        if mm:
            plt.axvline(x=t.mean(df2, loc=0, scale=1), color='y', ls="-.", label='Mean and Median')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig4)

    elif opcao == "Binomial":
        st.header("Probability Mass Function:")
        n = round(st.sidebar.number_input('N Trials:', min_value=0, value=6, step=1))
        p = st.sidebar.number_input('Probability of Success:',
                                    min_value=0.0, max_value=1.0, value=0.4, step=0.1)
        x = range(0, n + 1)  # valores de x =0,1,2,3..ou n
        dados_binom = binom.pmf(x, n, p)  # distribuição dos resultados

        fig5, ax5 = plt.subplots()
        ax5.plot(x, dados_binom, "bD")
        ax5.set_ylabel("Probability Mass")
        ax5.vlines(x, 0, dados_binom, colors='blue', lw=7, alpha=0.3)
        ax5.vlines(x, 0, dados_binom, colors='k', linestyles='-.', lw=1)
        plt.hlines(xmin=-1, xmax=x, y=dados_binom, color="lightgray", linestyles='dotted', alpha=0.5)
        plt.axvline(x=n * p, color='y', ls="-.", label='Mean')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig5)

    elif opcao == "Poisson":
        st.header("Probability Mass Function:")
        n2 = st.sidebar.number_input('N total:', min_value=0, value=10, step=1)
        lambd2 = st.sidebar.number_input('Lambda/mean:', min_value=0.1, value=4.5, step=0.25)
        x1 = range(0, n2 + 1)
        dados_poisson = poisson.pmf(x1, mu=lambd2)

        fig6, ax6 = plt.subplots()
        ax6.plot(x1, dados_poisson, "bD")
        ax6.vlines(x1, 0, dados_poisson, colors='blue', lw=7, alpha=0.3)
        ax6.vlines(x1, 0, dados_poisson, colors='k', linestyles='-.', lw=1)
        ax6.set_ylabel("Probability Mass")
        plt.axvline(x=np.mean(lambd2), color='y', ls="-.", label='Mean')
        plt.hlines(xmin=-1, xmax=x1, y=dados_poisson, color="lightgray", linestyles='dotted', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig6)

elif inicio == "Bivariate Normal Distribution":
    st.header("Probability Density Function: Bivariate Normal Distribution")
    mu1 = st.sidebar.number_input('Mean1:', value=0.0, step=0.5)
    sd1 = st.sidebar.number_input('sd1:', min_value=0.0, value=1.0, step=0.5)
    mu2 = st.sidebar.number_input('Mean2:', value=0.0, step=0.5)
    sd2 = st.sidebar.number_input('sd2:', min_value=0.0, value=1.0, step=0.5)
    cor = st.sidebar.number_input('Corr:', min_value=-0.9, max_value=0.9, value=0.0, step=0.1)

    mu = [mu1, mu2]
    mcov = [[(sd1 ** 2), cor], [cor, (sd2 ** 2)]]
    x = np.linspace(-6, 6, 500)
    y = np.linspace(-6, 6, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(cov=mcov, mean=mu)

    fig = make_subplots()
    fig.add_trace(go.Surface(x=X, y=Y, z=rv.pdf(pos),
                             colorscale='Viridis', showscale=False))

    st.plotly_chart(fig, use_container_width=True)

    fig2 = plt.figure()
    ax2 = plt.subplot()
    ax2.contourf(Y, X, rv.pdf(pos))
    ax2.set_xlabel("X 2")
    ax2.set_ylabel("X 1")
    st.pyplot(fig2)

elif inicio == "Central Limit Theorem":
    opcao = st.sidebar.selectbox("Distribution", ("Uniform", "Bernoulli",
                                                  "Poisson", "Exponential", "Cauchy"))

    if opcao == "Uniform":
        numero0 = st.sidebar.number_input('Number of Samples:', min_value=1, value=2, step=5)
        st.sidebar.write("1000 random numbers will be generated between:")
        values = st.sidebar.slider('Select a range of values to sample', -100, 100, (-10, 10))
        media_un = (values[0] + values[1]) / 2
        means = []
        np.random.seed(999)

        for i in range(numero0):
            rng = np.random.default_rng()
            media = np.mean(rng.integers(low=values[0], high=values[1] + 1, size=100))
            means.append(media)

        x1 = np.linspace(norm.ppf(0.001, media_un, np.std(means)),
                         norm.ppf(0.999, media_un, np.std(means)), 1000)
        y_nor = norm.pdf(x1, media_un, np.std(means))

        fig0, ax0 = plt.subplots()
        ax0.hist(means, density=True, bins="auto")
        ax0.plot(x1, y_nor, "r--", alpha=0.5, label='Normal')
        ax0.set_ylabel("Density")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig0)

    if opcao == "Bernoulli":
        numero = st.sidebar.number_input('Number of Samples:', min_value=1, value=2, step=5)
        p = st.sidebar.number_input('Probability:', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        d = bernoulli(p)
        res = []
        media = numero * p

        for i in range(1000):
            res.append(d.rvs(numero).sum())

        fig, ax = plt.subplots()
        nn, bins, empty = ax.hist(res, bins='auto', density=True)
        x1 = np.linspace(norm.ppf(0.001, media, np.std(res)),
                         norm.ppf(0.999, media, np.std(res)), 1000)
        y_nor = norm.pdf(x1, media, np.std(res))
        ax.plot(x1, y_nor, "r--", alpha=0.5, label='Normal')
        ax.set_ylabel("Density")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        fig.tight_layout()
        st.pyplot(fig)
    if opcao == "Poisson":
        numero2 = st.sidebar.number_input('Number of Samples:', min_value=1, value=2, step=5)
        lambd = st.sidebar.number_input('Lambda:', min_value=0.1, value=5.0, step=0.25)
        d = poisson(lambd)
        res = []

        for i in range(numero2):
            res.append((d.rvs(size=numero2).mean()))

        x1 = np.linspace(norm.ppf(0.001, np.mean(res), np.std(res)),
                         norm.ppf(0.999, np.mean(res), np.std(res)), 1000)
        y_nor = norm.pdf(x1, np.mean(res), np.std(res))

        fig2, ax2 = plt.subplots()
        ax2.hist(res, bins="auto", density=True)
        ax2.plot(x1, y_nor, "r--", alpha=0.5, label='Normal')
        ax2.set_ylabel("Density")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        fig2.tight_layout()
        st.pyplot(fig2)

    if opcao == "Exponential":
        numero3 = st.sidebar.number_input('Number of Samples:', min_value=1, value=2, step=5)
        lambd2 = st.sidebar.number_input('Lambda:', min_value=0.1, value=5.0, step=0.25)
        e = expon(lambd2)
        ss = []

        for i in range(numero3):
            ss.append(e.rvs(size=numero3).mean())

        x1 = np.linspace(norm.ppf(0.001, np.mean(ss), np.std(ss)),
                         norm.ppf(0.999, np.mean(ss), np.std(ss)), 1000)
        y_nor = norm.pdf(x1, np.mean(ss), np.std(ss))

        fig3, ax3 = plt.subplots()
        ax3.hist(ss, density=True, bins="auto")
        ax3.plot(x1, y_nor, "r--", alpha=0.5, label='Normal')
        ax3.set_ylabel("Density")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig3)

    if opcao == "Cauchy":
        numero4 = st.sidebar.number_input('Number of Samples:', min_value=1, value=2, step=5)
        par1 = st.sidebar.number_input('Parameter 1:', value=0.0, step=1.0)
        par2 = st.sidebar.number_input('Parameter 2:', value=1.0, step=1.0)
        e = cauchy(par1, par2)
        cc = []

        for i in range(100):
            cc.append(e.rvs(size=numero4).mean())

        x1 = np.linspace(norm.ppf(0.001, np.mean(cc), np.std(cc)),
                         norm.ppf(0.999, np.mean(cc), np.std(cc)), 1000)
        y_nor = norm.pdf(x1, np.mean(cc), np.std(cc))

        fig4, ax4 = plt.subplots()
        ax4.hist(cc, density=True, bins="auto")
        ax4.plot(x1, y_nor, "r--", alpha=0.5, label='Normal')
        ax4.set_ylabel("Density")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        st.pyplot(fig4)

elif inicio == "Quantiles":
    opcao = st.sidebar.selectbox("Distribution", ("Normal",
                                                  "Chi-squared", "F", "Binomial"))

    if opcao == "Normal":
        st.sidebar.subheader("N ~ (0,1)")
        q = st.sidebar.slider('Select the q', 0.001, 0.999, 0.25)
        x = np.linspace(norm.ppf(0.00001),
                        norm.ppf(0.99999), 1000)
        x1 = []

        for i in x:
            if i < norm.ppf(q):
                x1.append(i)

        y1 = list(norm.pdf(x)[:len(x1)])
        y1.append(0)
        x1.append(norm.ppf(q))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=norm.pdf(x), mode='lines', name="Area = 1-q",
                                 fillcolor="blue", fill="toself",
                                 fillpattern=dict(shape="-", fgopacity=0.5)))
        fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', fill="toself", fillcolor="magenta", name="Area = q"))
        fig.add_trace(go.Scatter(x=[0, x1[-1]], y=[0, 0], mode="markers+text", name="Xq",
                                 text=["", round(x1[-1], 3)], textposition="bottom center",
                                 textfont=dict(size=15)))
        fig.update_xaxes(zerolinewidth=0.5, zerolinecolor='LightPink')
        fig.add_annotation(x=-2.7, y=0.05, text="Area = q", showarrow=True, arrowcolor="magenta",
                           arrowhead=1, ax=-60, ay=-50, font=dict(size=24, color="magenta"))
        fig.add_annotation(x=2.7, y=0.1, text="Area = 1-q", showarrow=True, arrowcolor="blue",
                           arrowhead=1, ax=60, ay=-30, font=dict(size=24, color="blue"))
        fig.update_layout(title="Probability Density Function",
                          legend=dict(borderwidth=2, font=dict(size=15)))
        st.plotly_chart(fig, theme=None)

        st.sidebar.write("q = ", q, "Xq = ", round(x1[-1], 3))
        xq = norm.ppf(q)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=norm.cdf(x), name="Cumulative Density Function",
                                  mode='lines'))
        fig2.add_trace(go.Scatter(x=[xq, xq], y=[0, norm.cdf(xq)], mode="lines+markers+text",
                                  name="X axis: Xq", text=[round(xq, 3), ""],
                                  textposition="bottom center",
                                  line=dict(color="blue", dash="dot"),
                                  textfont=dict(size=15), marker=dict(color="green")))
        fig2.add_trace(go.Scatter(x=[xq, 0], y=[norm.cdf(xq), q], mode="lines+markers+text",
                                  name="Y axis: q", text=["", q], textposition="bottom center",
                                  line=dict(color="blue", dash="dot"), textfont=dict(size=15),
                                  marker=dict(color="magenta")))
        fig2.update_xaxes(zerolinewidth=0.5, zerolinecolor='LightPink')
        fig2.update_layout(title="Cumulative Density Function",
                           legend=dict(borderwidth=2, font=dict(size=15)))
        st.plotly_chart(fig2, theme=None)

    if opcao == "Chi-squared":
        q2 = st.sidebar.slider('Select the q', 0.001, 0.999, 0.25)
        df = st.sidebar.slider('Degress of freedom', 1, 10, 3)
        x = np.linspace(chi2.ppf(0.0000000001, df),
                        chi2.ppf(0.9999999999, df), 1000)
        x1 = []

        for i in x:
            if i < chi2.ppf(q2, df):
                x1.append(i)

        y1 = list(chi2.pdf(x, df)[:len(x1)])
        y1.append(0)
        x1.append(chi2.ppf(q2, df))
        y1[0] = 0
        x1[0] = 0

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=chi2.pdf(x, df), mode='lines', name="Area = 1-q",
                                  fillcolor="blue", fill="tozeroy",
                                  fillpattern=dict(shape="-", fgopacity=0.5)))
        fig2.add_trace(go.Scatter(x=x1, y=y1, mode='lines', fill="toself", fillcolor="magenta", name="Area = q"))
        fig2.add_trace(go.Scatter(x=[0, x1[-1]], y=[0, 0], mode="markers+text", name="Xq",
                                  text=["", round(x1[-1], 3)], textposition="bottom center",
                                  textfont=dict(size=15)))
        fig2.update_xaxes(range=[-3, 30])
        fig2.update_yaxes(range=[-0.1, 0.5])
        fig2.update_xaxes(zerolinewidth=0.5, zerolinecolor='LightPink')

        if df < 3:
            fig2.add_annotation(x=1.8, y=0.35, text="Area = q", showarrow=True, arrowcolor="magenta",
                                arrowhead=1, ax=60, ay=-50, font=dict(size=24, color="magenta"))
            fig2.add_annotation(x=10, y=0.06, text="Area = 1-q", showarrow=True, arrowcolor="blue",
                                arrowhead=1, ax=80, ay=-15, font=dict(size=24, color="blue"))
        elif 3 <= df <= 5:
            fig2.add_annotation(x=2.0, y=0.3, text="Area = q", showarrow=True, arrowcolor="magenta",
                                arrowhead=1, ax=8, ay=-60, font=dict(size=24, color="magenta"))
            fig2.add_annotation(x=10, y=0.05, text="Area = 1-q", showarrow=True, arrowcolor="blue",
                                arrowhead=1, ax=80, ay=-18, font=dict(size=24, color="blue"))
        else:
            fig2.add_annotation(x=3.8, y=0.2, text="Area = q", showarrow=True, arrowcolor="magenta",
                                arrowhead=1, ax=10, ay=-60, font=dict(size=24, color="magenta"))
            fig2.add_annotation(x=19, y=0.06, text="Area = 1-q", showarrow=True, arrowcolor="blue",
                                arrowhead=1, ax=90, ay=-20, font=dict(size=24, color="blue"))

        fig2.update_layout(title="Probability Density Function",
                           legend=dict(borderwidth=2, font=dict(size=15)))
        st.plotly_chart(fig2, theme=None)

        st.sidebar.write("q = ", q2, "Xq = ", round(x1[-1], 3))
        xq = chi2.ppf(q2, df)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x, y=chi2.cdf(x, df), name="Cumulative Density Function", mode='lines'))
        fig3.add_trace(go.Scatter(x=[xq, xq], y=[0, chi2.cdf(xq, df)], mode="lines+markers+text",
                                  name="X axis: Xq", text=[round(xq, 3), ""],
                                  textposition="bottom center", line=dict(color="blue", dash="dot"),
                                  textfont=dict(size=15), marker=dict(color="green")))
        fig3.add_trace(go.Scatter(x=[xq, 0], y=[chi2.cdf(xq, df), q2], mode="lines+markers+text",
                                  name="Y axis: q", text=["", q2], textposition="bottom center",
                                  line=dict(color="blue", dash="dot"), textfont=dict(size=15),
                                  marker=dict(color="magenta")))
        fig3.update_xaxes(zerolinewidth=0.5, zerolinecolor='LightPink')
        fig3.update_layout(title="Cumulative Density Function",
                           legend=dict(borderwidth=2, font=dict(size=15)))
        st.plotly_chart(fig3, theme=None)

    if opcao == "F":
        q3 = st.sidebar.slider('Select the q', 0.001, 0.999, 0.25)
        df1 = st.sidebar.slider('Degress of freedom 1', 10, 60, 30)
        df2 = st.sidebar.slider('Degress of freedom 2', 10, 60, 30)
        x = np.linspace(f.ppf(0.0000000001, df1, df2),
                        f.ppf(0.9999999999, df1, df2), 3000)
        x1 = []

        for i in x:
            if i < f.ppf(q3, df1, df2):
                x1.append(i)

        y1 = list(f.pdf(x, df1, df2)[:len(x1)])
        y1.append(0)
        x1.append(f.ppf(q3, df1, df2))
        y1[0] = 0
        x1[0] = 0

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=f.pdf(x, df1, df2), mode='lines', name="Area = 1-q",
                                  fillcolor="blue", fill="tozeroy",
                                  fillpattern=dict(shape="-", fgopacity=0.5)))
        fig2.add_trace(go.Scatter(x=x1, y=y1, mode='lines', fill="toself", fillcolor="magenta", name="Area = q"))
        fig2.add_trace(go.Scatter(x=[0, x1[-1]], y=[0, 0], mode="markers+text", name="Xq",
                                  text=["", round(x1[-1], 3)], textposition="bottom center",
                                  textfont=dict(size=15)))
        fig2.update_xaxes(range=[-1, 10])
        fig2.update_yaxes(range=[-0.05, 1.5])
        fig2.update_xaxes(zerolinewidth=0.5, zerolinecolor='LightPink')
        fig2.add_annotation(x=2, y=0.9, text="Area = q", showarrow=True, arrowcolor="magenta",
                            arrowhead=1, ax=60, ay=-50, font=dict(size=24, color="magenta"))
        fig2.add_annotation(x=4.8, y=0.1, text="Area = 1-q", showarrow=True, arrowcolor="blue",
                            arrowhead=1, ax=80, ay=-15, font=dict(size=24, color="blue"))
        fig2.update_layout(title="Probability Density Function",
                           legend=dict(borderwidth=2, font=dict(size=15)))
        st.plotly_chart(fig2, theme=None)

        st.sidebar.write("q = ", q3, "Xq = ", round(x1[-1], 3))
        xq = f.ppf(q3, df1, df2)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x, y=f.cdf(x, df1, df2), name="Cumulative Density Function", mode='lines'))
        fig3.add_trace(go.Scatter(x=[xq, xq], y=[0, f.cdf(xq, df1, df2)], mode="lines+markers+text",
                                  name="X axis: Xq", text=[round(xq, 3), ""],
                                  textposition="bottom center", line=dict(color="blue", dash="dot"),
                                  textfont=dict(size=15), marker=dict(color="green")))
        fig3.add_trace(go.Scatter(x=[xq, 0], y=[f.cdf(xq, df1, df2), q3], mode="lines+markers+text",
                                  name="Y axis: q", text=["", q3], textposition="bottom center",
                                  line=dict(color="blue", dash="dot"), textfont=dict(size=15),
                                  marker=dict(color="magenta")))
        fig3.update_xaxes(zerolinewidth=0.5, zerolinecolor='LightPink', range=[-1, 10])
        fig3.update_layout(title="Cumulative Density Function",
                           legend=dict(borderwidth=2, font=dict(size=15)))
        st.plotly_chart(fig3, theme=None)

    if opcao == "Binomial":
        q4 = st.sidebar.slider('Select the q', 0.0, 1.0, 0.5)
        n = st.sidebar.slider('Select the n', 1, 10, 5)
        p = st.sidebar.slider('Select the p', 0.01, 0.99, 0.50)
        x = list(range(0, n + 1))
        Xq = int(binom.ppf(q4, n, p))
        x1 = list(range(0, Xq + 1))
        y1 = binom.pmf(x1, n, p)
        x2 = list(range(Xq + 1, n + 1))
        y2 = binom.pmf(x2, n, p)

        fig = go.Figure()
        fig.add_trace(go.Bar(y=y1, x=x1, width=0.01, showlegend=False, opacity=0.7))
        fig.add_trace(go.Scatter(x=x1, y=y1, mode="markers",
                                 name="Sum = q", marker=dict(size=14, symbol="diamond", color="magenta")))
        fig.add_trace(go.Scatter(x=[Xq, 0], y=[0, 0], mode="markers+text", text=[Xq, ""],
                                 showlegend=True, textposition=["bottom center", "bottom center"],
                                 textfont=dict(size=15), name="Xq",
                                 marker=dict(size=[12, 1], symbol=["diamond", "diamond"],
                                             color=["green", "green"])))
        fig.add_trace(go.Bar(y=y2, x=x2, width=0.01, showlegend=False, opacity=0.1))
        fig.update_layout(title="Probability Mass Function",
                          legend=dict(borderwidth=1, font=dict(size=15), itemsizing="constant"))
        # fig.update_layout(legend={'itemsizing': 'constant'})
        st.plotly_chart(fig, theme=None)

        xt = list(range(-1, n + 1))
        proba = list(binom.pmf(xt, n, p))

        fig2 = px.ecdf(y=proba, x=xt, markers=True, ecdfmode="standard", opacity=0.6,
                       title="Cumulative Density Function", labels={"x": "Xq"})
        fig2.update_traces(marker=dict(size=13, symbol="circle", color="black", opacity=1))
        fig2.add_trace(go.Scatter(x=[xt[-1], xt[-1] + 1], y=[1, 1], mode="lines", showlegend=False,
                                  line=dict(color="black")))
        fig2.add_trace(go.Scatter(x=x, y=binom.cdf(range(-1, n), n, p), mode="markers",
                                  marker=dict(size=15, symbol="circle-open", color="black"),
                                  showlegend=False))
        fig2.add_trace(go.Scatter(x=[0, Xq], y=[q4, q4], mode="lines+markers+text", name="q",
                                  text=[q4, ""], textposition=["middle left", "top left"],
                                  textfont=dict(size=15), line=dict(color="blue", dash="dot"),
                                  marker=dict(size=(18, 8), symbol=("diamond", "x-open-dot"),
                                              color=("magenta", "blue"))))
        fig2.add_trace(go.Scatter(x=[Xq, Xq], y=[0, q4], mode="lines+markers+text", name="Xq",
                                  text=[Xq, ""], textposition=["bottom center", "top left"],
                                  textfont=dict(size=15), line=dict(color="blue", dash="dot"),
                                  marker=dict(size=(18, 8), symbol=("diamond", "x-open-dot"),
                                              color=("green", "blue"))))
        fig2.update_yaxes(range=[-0.3, 1.3])
        fig2.update_layout(legend=dict(borderwidth=1, font=dict(size=15)))
        st.plotly_chart(fig2, theme=None)
