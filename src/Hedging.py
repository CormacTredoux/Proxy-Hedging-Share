# Hedging strategies
# Cormac Tredoux 2025


from Pricing import Price, Price_F
import numpy as np
import os




#Non-Self_financing Proxy Hedger
def ProxyHedge(Strike, assets, weights, sigma_annual, corr, r, T, Price, sigma_fund, fund):
    tracking_errors = np.zeros(T)
    Hedge = np.zeros(T)
    V = np.zeros(T)
    grads = np.zeros(T)
    
    # catch static vs dynamic weights
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        # Static weights: repeat same vector for all t
        weights = np.tile(weights, (T, 1))  
   


    V[0], grads[0] = Price_F(Strike, fund,  sigma_fund, r, T, 0)
    phi_0 = np.dot(grads[0], weights[0])

    Hedge[0] = V[0]

    for t in range(1, T):
        if t > T:
            break

        V[t], grads[t] = Price(Strike, assets, weights[t], sigma_annual, corr, r, T, t)
        phi_t = grads[t]*weights[t]

        delta_S = assets[:, t] - assets[:, t-1]
        delta_H = np.dot(phi_t, delta_S)

        Hedge[t] = Hedge[t-1] + delta_H
        tracking_errors[t] = Hedge[t] - Hedge[t-1] - (V[t] - V[t-1])

    return tracking_errors, Hedge, V, grads


# Self_financing Proxy Hedger  
def ProxyHedge_self(Strike, assets, weights, sigma_annual, corr, r, T, Price, sigma_fund, fund):
    YEAR_DAYS = 252.0
    DT = 1.0 / YEAR_DAYS

    d, TT = assets.shape  
    assert TT >= T, "assets length must cover horizon T"

    tracking_errors = np.zeros(T)
    Hedge = np.zeros(T)
    V = np.zeros(T)
    grads = np.zeros(T)   

    # catch dynamic vs static weights
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        # Static weights: repeat same vector for all t
        weights = np.tile(weights, (T, 1))  

    
    V[0], _ = Price_F(Strike, fund, sigma_fund, r, T, 0)

    # get the per-asset gradient at t=0 
    _, g0 = Price(Strike, assets, weights[0], sigma_annual, corr, r, T, 0)
    grads[0] = g0

    # Holdings per asset at t=0 
    phi_prev = grads[0] * weights[0]        
    S0 = assets[:, 0]

    # Cash chosen so that hedge value equals liability at t=0 
    cash = V[0] - np.dot(phi_prev, S0)
    Hedge[0] = np.dot(phi_prev, S0) + cash

   
    for t in range(1, T):
        if t > T:
            break

        # Liability & per-asset gradient 
        V[t], grads[t] = Price(Strike, assets, weights[t], sigma_annual, corr, r, T, t)
        phi_t = grads[t] * weights[t]       

        S_t   = assets[:, t]
        S_tm1 = assets[:, t-1]

        # cash accrues interest
        cash *= (1.0 + r * DT)

        #  asset P&L with previous holdings
        delta_S = S_t - S_tm1
        dH_assets = np.dot(phi_prev, delta_S)

        
        

        #  rebalance to phi_t at current prices 
        trade = phi_t - phi_prev
        cost  = np.dot(trade, S_t)
        cash -= cost

        # final hedge value
        Hedge[t] = np.dot(phi_t, S_t) + cash

        
        tracking_errors[t] = (Hedge[t] - Hedge[t-1]) - (V[t] - V[t-1])

        # roll positions
        phi_prev = phi_t

    return tracking_errors, Hedge, V, grads


#Non-Self-Financing Hedger
def Hedge_F(Strike, fund, sigma, r, T, Price_F):
    tracking_errors = np.zeros(T)
    Hedge = np.zeros(T)
    V = np.zeros(T)
    grads = np.zeros(T)
    
    V[0], grads[0] = Price_F(Strike, fund,  sigma, r, T, 0)
    phi_0 = grads[0]
    
    Hedge[0] = V[0]
    
    for t in range(1, T):
        if t > T:
            break

        V[t], grads[t] = Price_F(Strike, fund, sigma, r, T, t)
        phi_t = grads[t]

        delta_S = fund[t] - fund[t-1]
        delta_H = phi_t * delta_S

        Hedge[t] = Hedge[t-1] + delta_H
        tracking_errors[t] = Hedge[t] - Hedge[t-1] - (V[t] - V[t-1])

    return tracking_errors, Hedge, V, grads




def Hedge_F_self(Strike, fund, sigma, r, T, Price_F):
    YEAR_DAYS = 252.0
    DT = 1.0 / YEAR_DAYS

    tracking_errors = np.zeros(T)
    Hedge = np.zeros(T)
    V = np.zeros(T)
    grads = np.zeros(T)  

    
    V[0], grads[0] = Price_F(Strike, fund, sigma, r, T, 0)
    phi_prev = grads[0]                 
    S0 = fund[0]

    # Cash chosen so hedge value equals liability at t=0
    cash = V[0] - phi_prev * S0
    Hedge[0] = phi_prev * S0 + cash

    
    for t in range(1, T):
        if t > T:
            break

        # Price liability and get new delta at time t
        V[t], grads[t] = Price_F(Strike, fund, sigma, r, T, t)
        phi_t = grads[t]               

        S_t   = fund[t]
        

        # cash accrues interest 
        cash *= (1.0 + r * DT)

        #  rebalance to phi_t at current price 
        trade = phi_t - phi_prev
        cost  = trade * S_t
        cash -= cost

        # final hedge value
        Hedge[t] = phi_t * S_t + cash

       
        tracking_errors[t] = (Hedge[t] - Hedge[t-1]) - (V[t] - V[t-1])

        # Roll position
        phi_prev = phi_t

    return tracking_errors, Hedge, V, grads