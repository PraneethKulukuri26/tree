from typing import Optional, Dict, List
# optimizer/quantum_optimizer.py
from django.http import JsonResponse
import numpy as np, pandas as pd, math, warnings
import matplotlib.pyplot as plt
import yfinance as yf

#from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from docplex.mp.model import Model

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)

# 👉 Paste all your helper functions here (Ledoit-Wolf, BL, QUBO builders, etc.)


def ledoit_wolf_shrinkage(Sigma_sample: np.ndarray, returns: np.ndarray):
    # returns: T x n matrix of returns (daily)
    n = Sigma_sample.shape[0]
    T = returns.shape[0]
    sample = Sigma_sample.copy()
    mu = np.diag(sample).mean()
    F = np.eye(n) * mu
    # compute pi_hat
    X = returns - returns.mean(axis=0, keepdims=True)
    pi_hat = 0.0
    for t in range(T):
        xt = X[t:t+1].T @ X[t:t+1]
        pi_hat += np.sum((xt - sample) ** 2)
    pi_hat /= T
    # rho_hat
    rho_hat = np.sum((sample - F) ** 2)
    # gamma
    gamma = np.linalg.norm(sample - F, 'fro')**2
    # alpha (shrinkage intensity)
    kappa = (pi_hat - rho_hat) / gamma if gamma > 0 else 0.0
    alpha = max(0.0, min(1.0, kappa / T))
    Sigma_shrunk = (1 - alpha) * sample + alpha * F
    return Sigma_shrunk, alpha


def black_litterman(mu_prior: np.ndarray, Sigma: np.ndarray, P: Optional[np.ndarray], Q: Optional[np.ndarray],
                    tau: float = 0.05, omega: Optional[np.ndarray] = None):
    # Simple BL posterior mean (returns)
    n = len(mu_prior)
    if P is None or Q is None:
        return mu_prior
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))
    A = np.linalg.pinv(tau * Sigma)
    middle = A + P.T @ np.linalg.pinv(omega) @ P
    rhs = A @ mu_prior + P.T @ np.linalg.pinv(omega) @ Q
    mu_bl = np.linalg.pinv(middle) @ rhs
    return mu_bl

def fetch_and_prepare(tickers, period='1y', interval='1d'):
    df = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how='all').ffill().dropna(axis=1)
    rets = df.pct_change().dropna()
    mu_d = rets.mean().values
    Sigma_d = rets.cov().values
    mu = mu_d * 252.0
    Sigma = Sigma_d * 252.0
    return df, rets.values, mu, Sigma



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


def binary_encoding_vars(n, bits_per_asset):
    # returns total bits and mapping asset->bit indices
    total_bits = n * bits_per_asset
    mapping = {i: list(range(i*bits_per_asset, (i+1)*bits_per_asset)) for i in range(n)}
    return total_bits, mapping

def weight_from_bits(y_bits: np.ndarray, mapping: Dict[int, List[int]], bits_per_asset: int, max_weight: float):
    # interpret bits as unsigned integer and scale to [0, max_weight]
    n = len(mapping)
    w = np.zeros(n)
    for i in range(n):
        inds = mapping[i]
        val = 0
        for b_idx, bitpos in enumerate(inds):
            val += int(y_bits[bitpos]) * (2**b_idx)
        max_int = 2**bits_per_asset - 1
        w[i] = (val / max_int) * max_weight if max_int>0 else 0.0
    return w


def build_qubo_selection(mu, Sigma, rf, K, t_param, A_card, sector_caps, A_sec, x_prev, lambda_ham, trans_cost):
    r = mu - rf
    n = len(r)
    mdl = Model('select_qubo')
    x = mdl.binary_var_list(n, name='x')

    lin = mdl.sum(r[i] * x[i] for i in range(n))
    quad_risk = mdl.sum(Sigma[i,j] * x[i] * x[j] for i in range(n) for j in range(n))
    obj = lin - (t_param / K) * quad_risk

    # cardinality
    sumx = mdl.sum(x)
    obj -= A_card * (sumx - K) * (sumx - K)

    # sector caps
    for name, (idxs, cap) in sector_caps.items():
        if len(idxs)==0: continue
        sx = mdl.sum(x[i] for i in idxs)
        obj -= A_sec * (sx - cap) * (sx - cap)

    # turnover penalty (Hamming distance) and transaction cost approximation
    if x_prev is not None:
        for i in range(n):
            prev = int(x_prev[i])
            # penalize changes: cost ~ lambda_ham*(x_i + prev - 2 x_i * prev) and tx cost proportional
            obj -= lambda_ham * (x[i] + prev - 2 * x[i] * prev)
            if not prev and trans_cost>0:
                # cost to buy: subtract trans_cost * expected weight approx (use equal weight approx)
                obj -= trans_cost * (1.0/K) * x[i]
    mdl.maximize(obj)
    qp = QuadraticProgram()
    qp = QuadraticProgram.from_docplex(mdl) if hasattr(QuadraticProgram, 'from_docplex') else QuadraticProgram()
    # safe convert via qiskit_optimization.translators if needed
    try:
        from qiskit_optimization.translators import from_docplex_mp
        qp = from_docplex_mp(mdl)
    except Exception:
        # fallback: directly build variables
        pass
    return qp

def build_qubo_binary_weights(mu, Sigma, rf, bits_per_asset, max_weight, t_param, A_budget, budget_val, sector_caps, A_sec, x_prev, lambda_ham, trans_cost):
    # Build QUBO over binary bits representing weights
    n = len(mu)
    total_bits, mapping = binary_encoding_vars(n, bits_per_asset)
    mdl = Model('binary_weight_qubo')
    y = mdl.binary_var_list(total_bits, name='y')

    # map bits to weights symbolically: weight_i = (sum 2^b y_b) / (2^L -1) * max_weight
    max_int = 2**bits_per_asset - 1
    scale = max_weight / max_int if max_int>0 else 0.0

    # linear return term: r^T w
    r = mu - rf
    lin = mdl.sum( r[i] * scale * mdl.sum( (2**(b_idx)) * y[b_idx_global] 
                                          for b_idx, b_idx_global in enumerate(mapping[i]) ) 
                  for i in range(n) )

    # quadratic risk term: w^T Sigma w -> quadratic in y bits
    quad = mdl.sum( (scale * (2**(b_idx)) * (scale * (2**(b_jdx)))) * Sigma[i,j] * y[b_idx_global] * y[b_jdx_global]
                    for i in range(n) for j in range(n)
                    for b_idx, b_idx_global in enumerate(mapping[i])
                    for b_jdx, b_jdx_global in enumerate(mapping[j]) )

    obj = lin - t_param * quad

    # budget constraint: sum weights approx == budget_val (like 1.0)
    # encode (sum_i w_i - budget_val)^2 penalty
    sumw = mdl.sum( scale * mdl.sum((2**b_idx) * y[b_idx_global] for b_idx, b_idx_global in enumerate(mapping[i])) for i in range(n))
    obj -= A_budget * (sumw - budget_val) * (sumw - budget_val)

    # sector caps: approximate by bounding sum of bits in each sector (coarse)
    for name, (idxs, cap) in sector_caps.items():
        # convert cap in weights -> cap_bits approx
        cap_bits = int(np.ceil(cap / max_weight * (2**bits_per_asset - 1))) * len(idxs)
        sx = mdl.sum(y[b] for i in idxs for b in mapping[i])
        obj -= A_sec * (sx - cap_bits) * (sx - cap_bits)

    # turnover and txcost approximations (not exact)
    if x_prev is not None:
        # penalize changes on most-significant bit as proxy
        for i in range(n):
            prev = int(x_prev[i])
            msb_idx = mapping[i][-1]
            obj -= lambda_ham * ( y[msb_idx] + prev - 2 * y[msb_idx] * prev )
    mdl.maximize(obj)
    try:
        from qiskit_optimization.translators import from_docplex_mp
        qp = from_docplex_mp(mdl)
    except Exception:
        qp = QuadraticProgram()
    return qp


def qaoa_solve_qp(qp: QuadraticProgram, p_list, shots, seed):
    qubo = QuadraticProgramToQubo().convert(qp)
    # sampler = Sampler(options={'seed':seed, 'shots':shots})  # Disabled
    best=None; best_res=None
    for p in p_list:
        qaoa = QAOA(sampler=sampler, reps=int(p), optimizer=SPSA(maxiter=120, blocking=True))
        solver = MinimumEigenOptimizer(qaoa)
        try:
            res = solver.solve(qubo)
        except Exception as e:
            print('QAOA solver failed:', e)
            continue
        # extract bits into vector
        x = np.array([int(res.variables_dict[k]) for k in sorted(res.variables_dict.keys(), key=lambda s: int(s.split('_')[-1]))])
        val = qubo.objective.evaluate(res.variables_dict)
        cand={'x':x,'p':p,'energy':val,'raw':res}
        if (best is None) or (val>best['energy']):
            best=cand; best_res=res
    return best, best_res

def dinkelbach_qaoa_runner(mu, Sigma, rf, CFG):
    use_discrete = CFG['use_discrete_weights']
    n = len(mu)
    # init t with greedy equal selection
    x0 = baseline_greedy(mu, Sigma, CFG['K'], rf)['x']
    w0 = equal_weights_from_x(x0)
    r = mu - rf
    t = float((r @ w0) / max(w0 @ Sigma @ w0, 1e-12))
    best_overall = None; history=[]
    lam_max = float(np.linalg.eigvalsh(Sigma).max())
    for it in range(CFG['dinkelbach_iters']):
        A_card = 10.0 * (t / CFG['K']) * lam_max
        A_sec = 5.0 * (t / CFG['K']) * lam_max
        if not use_discrete:
            qp = build_qubo_selection(mu, Sigma, rf, CFG['K'], t, A_card, CFG['sector_caps'], A_sec, CFG['prev_selection'], CFG['lambda_ham'], CFG['transaction_cost'])
        else:
            qp = build_qubo_binary_weights(mu, Sigma, rf, CFG['bits_per_asset'], CFG['max_weight'], t, A_card, 1.0, CFG['sector_caps'], A_sec, CFG['prev_selection'], CFG['lambda_ham'], CFG['transaction_cost'])
        best, raw = qaoa_solve_qp(qp, CFG['p_list'], CFG['shots'], CFG['seed'])
        if best is None:
            print('No valid QAOA result at iter', it); break
        x = best['x']
        # for discrete mode, map bits->weights; for selection mode, equal-weight then refine
        if not use_discrete:
            w_eq = equal_weights_from_x(x)
            numer = float((mu - rf) @ w_eq); denom = float(w_eq @ Sigma @ w_eq)
            s_eq = sharpe(w_eq, mu, Sigma, rf)
            t_new = numer / max(denom, 1e-12)
        else:
            # map bits to weights and compute sharpe
            total_bits, mapping = binary_encoding_vars(n, CFG['bits_per_asset'])
            w = weight_from_bits(x, mapping, CFG['bits_per_asset'], CFG['max_weight'])
            # normalize to sum to 1 if possible
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.zeros(n)
            w_eq = w; s_eq = sharpe(w_eq, mu, Sigma, rf)
            numer = float((mu - rf) @ w_eq); denom = float(w_eq @ Sigma @ w_eq)
            t_new = numer / max(denom, 1e-12)
        history.append({'iter':it, 't':t, 't_new':t_new, 'x':x, 'w_eq':w_eq, 'sharpe_eq':s_eq})
        if best_overall is None or s_eq > best_overall['sharpe_eq']:
            best_overall = history[-1]
        if abs(t_new - t) < 1e-3:
            t = t_new; break
        t = t_new
    return best_overall, history


def sharpe(weights, mu, Sigma, rf):
    port_ret = float(weights @ mu)
    port_var = float(weights @ Sigma @ weights)
    port_vol = math.sqrt(max(port_var, 1e-12))
    return (port_ret - rf) / (port_vol if port_vol>0 else 1e-12)

def equal_weights_from_x(x):
    idx = np.flatnonzero(x)
    if len(idx)==0: return None
    w = np.zeros_like(x, dtype=float)
    w[idx] = 1.0/len(idx)
    return w

def subset_tangency(mu, Sigma, rf, x):
    S = np.flatnonzero(x)
    if len(S)==0: return None
    Sigma_SS = Sigma[np.ix_(S,S)]; r_S = mu[S] - rf
    inv = np.linalg.pinv(Sigma_SS)
    w_S = inv @ r_S
    denom = np.ones(len(S)) @ w_S
    if abs(denom)<1e-12: return None
    w_S = w_S / denom
    w = np.zeros(len(mu)); w[S]=w_S
    return w

def baseline_greedy(mu, Sigma, K, rf, lamb=4.0):
    diag = np.diag(Sigma); r = mu - rf; scores = r - lamb * diag
    pick = np.argsort(scores)[::-1][:K]
    x = np.zeros_like(mu, dtype=int); x[pick]=1
    w = equal_weights_from_x(x); s = sharpe(w, mu, Sigma, rf)
    return {'name':'greedy','x':x,'w':w,'sharpe':s}

def baseline_tangency(mu, Sigma, rf):
    r = mu - rf; inv = np.linalg.pinv(Sigma); w = inv @ r
    denom = np.ones_like(r) @ w
    if abs(denom)<1e-12: return {'name':'tangency','w':None,'sharpe':float('nan')}
    w = w / denom; return {'name':'tangency','w':w,'sharpe':sharpe(w, mu, Sigma, rf)}

from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def optimize_portfolio(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        params = json.loads(request.body.decode('utf-8'))
        tickers = params.get('tickers', ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','AMD','CRM','AVGO'])
        period = params.get('period', '1y')
        interval = params.get('interval', '1d')
        risk_free = float(params.get('risk_free', 0.02))
        budget = params.get('budget',1000)
        K = int(params.get('K', 5))
        prices=[]

        df, rets_daily, mu, Sigma = fetch_and_prepare(tickers, period, interval)
        n = len(mu)

        if(params.get('shrink_leduit_wolf', True)):
            Sigma_sh, alpha = ledoit_wolf_shrinkage(Sigma, rets_daily)
            Sigma = Sigma_sh

        greedy_res = baseline_greedy(mu, Sigma, K, risk_free)
        tangency_res = baseline_tangency(mu, Sigma, risk_free)

        w_refined = greedy_res['w']
        s_refined = greedy_res['sharpe']

        weights = np.array(w_refined) / np.sum(w_refined)

        print("Fetching latest prices using yfinance...")
        for t in tickers:
            data = yf.download(t, period="5d", progress=False)
            # Try Adj Close, else fallback to Close
            if "Adj Close" in data.columns:
                latest_price = data["Adj Close"].iloc[-1]
            elif "Close" in data.columns:
                latest_price = data["Close"].iloc[-1]
            else:
                raise ValueError(f"No valid price column found for {t}")
            prices.append(float(latest_price))   # ensure plain float

        allocations = weights * budget

        shares, spent = [], []
        for alloc, price in zip(allocations, prices):
            num_shares = int(alloc // price)
            cost = num_shares * price
            shares.append(num_shares)
            spent.append(cost)

        total_spent = sum(spent)
        leftover_cash = budget - total_spent

        portfolio_json = {
            "Portfolio": [
                {
                    "Asset": tickers[i],
                    "Weight": round(float(weights[i]), 4),
                    "Price($)": round(float(prices[i]), 2),
                    "Allocation($)": round(float(allocations[i]), 2),
                    "Shares": int(shares[i]),
                    "Spent($)": round(float(spent[i]), 2)
                }
                for i in range(len(tickers))
            ],
            "Summary": {
                "Total Spent($)": round(float(total_spent), 2),
                "Leftover Cash($)": round(float(leftover_cash), 2),
                "Budget($)": budget
            }
        }

        return JsonResponse(portfolio_json)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)