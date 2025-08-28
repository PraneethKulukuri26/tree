# optimizer/quantum_optimizer.py
import numpy as np, pandas as pd, math, warnings
import matplotlib.pyplot as plt
import yfinance as yf

from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from docplex.mp.model import Model

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)

# 👉 Paste all your helper functions here (Ledoit-Wolf, BL, QUBO builders, etc.)

def run_portfolio_optimizer(cfg: dict):
    """Main entry: takes a CFG dict, returns results"""
    prices, rets_daily, mu, Sigma = fetch_and_prepare(cfg['tickers'], cfg['period'], cfg['interval'])

    if cfg.get('shrink_leduit_wolf', False):
        Sigma, _ = ledoit_wolf_shrinkage(Sigma, rets_daily)

    greedy_res = baseline_greedy(mu, Sigma, cfg['K'], cfg['risk_free'])
    tangency_res = baseline_tangency(mu, Sigma, cfg['risk_free'])

    best_overall, history = dinkelbach_qaoa_runner(mu, Sigma, cfg['risk_free'], cfg)
    w_refined = subset_tangency(mu, Sigma, cfg['risk_free'], best_overall['x'])
    s_refined = sharpe(w_refined, mu, Sigma, cfg['risk_free'])

    results = {
        "greedy": greedy_res,
        "tangency": tangency_res,
        "qaoa_best": best_overall,
        "refined_sharpe": float(s_refined),
        "selected_tickers": [cfg['tickers'][i] for i in np.flatnonzero(best_overall['x']).tolist()],
        "weights": {cfg['tickers'][i]: float(w_refined[i]) for i in range(len(cfg['tickers'])) if w_refined[i] != 0}
    }
    return results
