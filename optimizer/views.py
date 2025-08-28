from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import pandas as pd
import math
import warnings
import yfinance as yf

def ledoit_wolf_shrinkage(Sigma_sample, returns):
	n = Sigma_sample.shape[0]
	T = returns.shape[0]
	sample = Sigma_sample.copy()
	mu = np.diag(sample).mean()
	F = np.eye(n) * mu
	X = returns - returns.mean(axis=0, keepdims=True)
	pi_hat = 0.0
	for t in range(T):
		xt = X[t:t+1].T @ X[t:t+1]
		pi_hat += np.sum((xt - sample) ** 2)
	pi_hat /= T
	rho_hat = np.sum((sample - F) ** 2)
	gamma = np.linalg.norm(sample - F, 'fro')**2
	kappa = (pi_hat - rho_hat) / gamma if gamma > 0 else 0.0
	alpha = max(0.0, min(1.0, kappa / T))
	Sigma_shrunk = (1 - alpha) * sample + alpha * F
	return Sigma_shrunk, alpha

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

def baseline_greedy(mu, Sigma, K, rf, lamb=4.0):
	diag = np.diag(Sigma); r = mu - rf; scores = r - lamb * diag
	pick = np.argsort(scores)[::-1][:K]
	x = np.zeros_like(mu, dtype=int); x[pick]=1
	w = equal_weights_from_x(x); s = sharpe(w, mu, Sigma, rf)
	return {'name':'greedy','x':x,'w':w,'sharpe':s}

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
		K = int(params.get('K', 5))
		shrink_leduit_wolf = bool(params.get('shrink_leduit_wolf', True))
		# Fetch data
		prices, rets_daily, mu, Sigma = fetch_and_prepare(tickers, period, interval)
		# Apply Ledoit-Wolf shrinkage if enabled
		if shrink_leduit_wolf:
			Sigma, alpha = ledoit_wolf_shrinkage(Sigma, rets_daily)
		# Run greedy baseline
		greedy_res = baseline_greedy(mu, Sigma, K, risk_free)
		sel = np.flatnonzero(greedy_res['x']).tolist()
		weights = greedy_res['w'].tolist() if greedy_res['w'] is not None else []
		sharpe_val = greedy_res['sharpe']
		result = {
			'selected_indices': sel,
			'selected_tickers': [tickers[i] for i in sel],
			'weights': weights,
			'sharpe': sharpe_val,
			'mu': mu.tolist(),
			'Sigma': Sigma.tolist(),
		}
		return JsonResponse(result)
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)
