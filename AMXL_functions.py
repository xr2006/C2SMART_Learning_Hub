#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:33:39 2025

@author: ryan
"""
import numpy as np
import pandas as pd
import cvxpy as cp

def solve_agent_commuting(
        aa: pd.DataFrame,
        initialize: np.ndarray,
        epsilon_c: np.ndarray,
        iid: int,
        safe_boundary: float
    ):
   
    # --- pull attributes of the chosen alternative -------------------------
    t_commute = float(aa.loc[aa['chosen'], 't_commute'])
    c_commute = float(aa.loc[aa['chosen'], 'c_commute'])
    M_commute = float(aa.loc[aa['chosen'], 'M_commute2'])
    SDE_work  = float(aa.loc[aa['chosen'], 'SDE_work'])
    SDL_work  = float(aa.loc[aa['chosen'], 'SDL_work'])
    PL_work   = float(aa.loc[aa['chosen'], 'PL_work'])
    ln_dwork  = float(aa.loc[aa['chosen'], 'ln_dwork'])
    aa = aa.reset_index(drop=True)
    chosen_i = int(aa.index[aa['chosen']].item())  # 0‑based row index

    # --- decision variables -------------------------------------------------
    x = cp.Variable(7)          # unconstrained continuous variables

    # --- constraints --------------------------------------------------------
    constraints = []
    for j in range(len(aa)):
        if aa.at[j, 'alternative'] != aa.loc[aa['chosen'], 'alternative'].values[0]:
            lhs = (
                (t_commute - aa.at[j, 't_commute'])   * x[0] +
                (c_commute - aa.at[j, 'c_commute'])   * x[1] +
                (M_commute - aa.at[j, 'M_commute2'])  * x[2] +
                (SDE_work  - aa.at[j, 'SDE_work'])    * x[3] +
                (SDL_work  - aa.at[j, 'SDL_work'])    * x[4] +
                (PL_work   - aa.at[j, 'PL_work'])     * x[5] +
                (ln_dwork  - aa.at[j, 'ln_dwork'])    * x[6]
            )
            rhs = (
                epsilon_c[iid - 1, j]
                - epsilon_c[iid - 1, chosen_i]
                + safe_boundary
            )
            constraints.append(lhs >= rhs)

    # --- objective: minimise distance to initial guess ----------------------
    objective = cp.Minimize(cp.sum_squares(x - initialize))

    # --- solve --------------------------------------------------------------
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)   # any conic solver (ECOS, OSQP, SCS…)
        variables = np.asarray(x.value).flatten()
        Z = prob.value
    except:                                       # infeasible, unbounded, or failed
        variables = np.zeros(7)
        Z = 0.0

    return variables, Z


def One_iteration_AMXL(Commuting_choice_ms,shuffle,epsilon_c, x_k, sample_size,bound, boundary_max,boundary_min,step):
    ##0.get initial parameters
    x_k_c = x_k

    ##1.Run three IO models
    parameter_i_lst_c = []
    sb_c = []
    for i in shuffle[:sample_size]:
        #commuting
        aa = Commuting_choice_ms[Commuting_choice_ms['iid']==i]
        safe_boundary_c = boundary_max
        parameter_i_c,Z = solve_agent_commuting(aa,x_k_c,epsilon_c,i,safe_boundary_c)
        while ((parameter_i_c.max()>bound)or(parameter_i_c.min()<-bound)or(parameter_i_c.sum()==0)) and (safe_boundary_c>boundary_min):
            safe_boundary_c -= step
            parameter_i_c,Z = solve_agent_commuting(aa,x_k_c,epsilon_c,i,safe_boundary_c)
        parameter_i_lst_c.append(parameter_i_c)
        sb_c.append(safe_boundary_c)
    
    #commuting
    parameter_i_lst_0_c = np.array(parameter_i_lst_c)
    parameter_i_lst_c = parameter_i_lst_0_c[(parameter_i_lst_0_c.max(axis=1)<bound)&
                                      (parameter_i_lst_0_c.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_c.sum(axis=1)!=0)]
    
    ##4.Calculate y_0
    y_k = parameter_i_lst_c.mean(axis=0)
    
    return y_k, parameter_i_lst_0_c,sb_c


def group_level_IO(Y_line, X_line, beta_0, lb, ub, tol):
    beta_0 = np.asarray(beta_0, dtype=float)
    n_beta = beta_0.size

    # ------------------------------------------------------------------
    # Decision variables and bounds
    # ------------------------------------------------------------------
    x = cp.Variable(n_beta)

    constraints = []
    if np.isfinite(lb).any():
        constraints.append(x >= lb)
    if np.isfinite(ub).any():
        constraints.append(x <= ub)

    # ------------------------------------------------------------------
    # Log‑share constraints
    # ------------------------------------------------------------------
    Y_line_ = np.maximum(Y_line, 0.001)          # avoid log(0)
    log_Y   = np.log(Y_line_)

    # pre‑compute V = X_line.T @ x  (but in CVXPY we build it explicitly)
    V = X_line.T @ x                 # shape (6,)

    for j in range(6):
        for k in range(6):
            if j != k and Y_line_[j] >= 0.001 and Y_line_[k] >= 0.001:
                rhs = log_Y[j] - log_Y[k]
                constraints += [
                    V[j] - V[k] <= rhs + float(tol),
                    V[j] - V[k] >= rhs - float(tol),
                ]

    # ------------------------------------------------------------------
    # Objective: minimise ||x − beta_0||²
    # ------------------------------------------------------------------
    w = np.ones_like(beta_0, dtype=float)
    obj_expr = cp.sum(cp.multiply(w, cp.square(x - beta_0)))
    objective = cp.Minimize(obj_expr)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)    # OSQP / SCS also work
        beta = x.value
        Z = prob.value
    except:                          # infeasible, unbounded, or failed
        beta = beta_0.copy()
        Z = 0.0

    # ------------------------------------------------------------------
    # Post‑fit metrics (same helper you already have)
    # ------------------------------------------------------------------
    rho, mse, mae, LL, LL_0, P = compute_metrics(Y_line, X_line, beta)
    beta[0]*=2e9
    beta[6]*=1e8
    beta[[0,6]] = np.array([-x if x>0 else x for x in beta[[0,6]]])
    beta[[0,6]] = np.array([-(-x%10) if x<-10 else x for x in beta[[0,6]]])

    return beta, Z, rho, mse, mae, LL, LL_0, P

def compute_metrics(Y_line,X_line,beta):
    V = (X_line*beta[:,None]).sum(axis=0)
    demo = np.exp(V).sum()
    P = np.exp(V) / demo
    LL = np.sum(Y_line * np.log(P))
    P_0 = np.zeros(6)+1/6
    LL_0 = np.sum(Y_line * np.log(P_0))
    rho = 1-LL/LL_0
    mse = np.mean(np.square(P-Y_line))
    mae = np.mean(np.abs(P-Y_line))
    return rho,mse,mae,LL,LL_0,P


