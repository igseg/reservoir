import pandas as pd
import numpy  as np
from os import listdir
from os.path import isfile, join
import re
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import data_tools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import least_squares
from scipy import stats
import statsmodels.api as sm
import calendar
from my_time_series import (
    fit_AR_LS,
    residuals_AR,
    tests_gaussian_white_noise,
)

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

import data_tools
import itertools

from matplotlib import cm
from scipy.stats import norm
from Models.models import *
from time import time
from mpl_toolkits.mplot3d import axes3d, Axes3D

from scipy.optimize import leastsq
import scipy.stats as spst
from numba import jit
from tabulate import tabulate
import scipy.fft

def saving_scale_results(policy = None, V0 = None, scale = None):
    policy.tofile(f'Results/policy_2_mr_model_{scale}.txt')
    V0.tofile(f'Results/V0_2_mr_model_{scale}.txt')

def load_scale_results(scale):
    mr_file = open(f'Results/policy_2_mr_model_{scale}.txt')
    loaded_policy = np.fromfile(mr_file)
    loaded_policy = loaded_policy.reshape((S,N,M))
    mr_file.close()

    V0_mr = open(f'Results/V0_2_mr_model_{scale}.txt')
    loaded_V0 = np.fromfile(V0_mr)
    loaded_V0 = loaded_V0.reshape((S+1,N,M))
    V0_mr.close()

    return loaded_policy, loaded_V0

def get_middle_points(x):
    x = np.array(x)
    return (x[:-1] + x[1:])/2


def pi_fun(storage, inflow, t, P, q, model_inflow, model_evaporation):

    expected_storage = model_storage(model_inflow, inflow , model_evaporation, t, storage, q) # this may not be needed if forecast is done.
    return eta * q * ((z(storage, elev_stor) + z(expected_storage,elev_stor))/2 - zd) * P

def pi_call_0(storage, inflow, t, P, q):
    if storage > 2e4 or storage < 3e4: # this must be wrong
        return pi_fun(storage, inflow, t, P, q, model_inflow, model_evaporation)
    else:
        return -1e7

def prepro_coord_array(x,y):

    cond_storage = np.logical_or(x<storage_grid[0], x>storage_grid[-1])
    x[x<storage_grid[0]]  = storage_grid[0]
    x[x>storage_grid[-1]] = storage_grid[-1]

    cond_price = np.logical_or(y<price_grid[0], y>price_grid[-1])
    y[y<price_grid[0] ] = price_grid[0]
    y[y>price_grid[-1]] = price_grid[-1]
    #cond_price = np.logical_or(x<price_grid[0], x>price_grid[-1])

    return x,y, cond_storage, cond_price


def future_payments(V,A,shocks_storage, shocks_price, input_next_storage, input_next_price, price_grid, storage_grid,boundaries,ts_forecast):
    K = len(shocks_storage)
    J = len(shocks_price  )

    future_payment = np.zeros((K,J))

    future_storage = next_storage(*input_next_storage, shocks_storage)
    cond = np.logical_or(future_storage < boundaries[0], future_storage > boundaries[1])

    ###################

    future_price = next_price_cte(  *input_next_price  , shocks_price) # adjust the price model
#     future_price = next_price_rever(  *input_next_price  , shocks_price) # adjust the price model

    ################3

    cond = np.meshgrid(cond,future_price)[0]

    under_storage = False
    over_storage  = False

    if future_storage.min() < boundaries[0]:
        under_storage = True
        over_storage  = False

    elif future_storage.max() > boundaries[1]:
        under_storage = False
        over_storage  = True


    future_storage, future_price, cond_storage, cond_price = prepro_coord_array(future_storage, future_price)
    future_storage_upper, future_price_upper = coord_array_upper(future_storage, future_price,cond_storage, cond_price) # at this point future_price is the coord in V
    future_storage_lower, future_price_lower = coord_array_lower(future_storage, future_price)
    future_storage_closest, future_price_closest = coord_array(future_storage, future_price) # at this point future_price is the coord in V

    #print(future_price_closest)

    future_storage_upper[cond_storage] = future_storage_closest[cond_storage]
    future_price_upper[cond_price] = future_price_closest[cond_price]
    future_storage_exact, future_price_exact = coord_array_exact(future_storage, future_price)

    weight_storage = np.ones(K) - future_storage_exact + future_storage_lower
    weight_price   = np.ones(J) - future_price_exact   + future_price_lower

    W_storage = np.outer(weight_storage, np.ones(K))
    W_price   = np.outer(weight_price,   np.ones(J))
    W = (W_storage + W_price)/2

    future_storage_lower_cd ,future_price_lower_cd = np.meshgrid(future_storage_lower, future_price_lower)
    future_storage_upper_cd ,future_price_upper_cd = np.meshgrid(future_storage_upper, future_price_upper)

    future_value = W * V[future_storage_lower_cd, future_price_lower_cd] + (-W + 1) * V[future_storage_upper_cd, future_price_upper_cd]

    future_payment = np.multiply(future_value, A)

    return np.sum(future_payment[~cond]), under_storage, over_storage

def action_value(input_next_storage, input_next_price, V0, A, shocks_storage, shocks_prices, price_grid, storage_grid, boundaries, ts_forecast):

    s_next = (s+1)
    next_state_value, under_storage, over_storage = future_payments(V0[s_next],A, shocks_storage, shocks_prices,
                                                            input_next_storage, input_next_price,
                                                            price_grid, storage_grid, boundaries, ts_forecast)

    next_state_value = discount * next_state_value

    if under_storage or over_storage:
        return 0 + next_state_value
    else:
        q = input_next_storage[-2]
        return pi_call_0(storage, inflow, current_time, price, q) + next_state_value

def save_load_policy_V0(policy = None, V0 = None, save_results = False, load_results = False, mr = False, one_component = True):

    if save_results and policy.any() and V0.any():
        if one_component:
            if mr:
                policy.tofile('Results/policy_mr_model.txt')
                V0.tofile('Results/V0_mr_model.txt')

            else:
                policy.tofile('Results/policy_nonmr_model.txt')
                V0.tofile('Results/V0_nonmr_model.txt')
        else:
            if mr:
                policy.tofile('Results/policy_2_mr_model.txt')
                V0.tofile('Results/V0_2_mr_model.txt')

            else:
                policy.tofile('Results/policy_2_nonmr_model.txt')
                V0.tofile('Results/V0_2_nonmr_model.txt')

    if load_results:
        if one_component:

            if mr:
                mr_file = open("Results/policy_mr_model.txt")
                loaded_policy = np.fromfile(mr_file)
                loaded_policy = loaded_policy.reshape((S,N,M))
                mr_file.close()

                V0_mr = open("Results/V0_mr_model.txt")
                loaded_V0 = np.fromfile(V0_mr)
                loaded_V0 = loaded_V0.reshape((S+1,N,M))
                V0_mr.close()

            else:
                nonmr_file = open("Results/policy_nonmr_model.txt")
                loaded_policy = np.fromfile(nonmr_file)
                loaded_policy = loaded_policy.reshape((S,N,M))
                nonmr_file.close()

                V0_nonmr = open("Results/V0_nonmr_model.txt")
                loaded_V0 = np.fromfile(V0_nonmr)
                loaded_V0 = loaded_V0.reshape((S+1,N,M))
                V0_nonmr.close()

        else:
            if mr:
                mr_file = open("Results/policy_2_mr_model.txt")
                loaded_policy = np.fromfile(mr_file)
                loaded_policy = loaded_policy.reshape((S,N,M))
                mr_file.close()

                V0_mr = open("Results/V0_2_mr_model.txt")
                loaded_V0 = np.fromfile(V0_mr)
                loaded_V0 = loaded_V0.reshape((S+1,N,M))
                V0_mr.close()

            else:
                nonmr_file = open("Results/policy_2_nonmr_model.txt")
                loaded_policy = np.fromfile(nonmr_file)
                loaded_policy = loaded_policy.reshape((S,N,M))
                nonmr_file.close()

                V0_nonmr = open("Results/V0_2_nonmr_model.txt")
                loaded_V0 = np.fromfile(V0_nonmr)
                loaded_V0 = loaded_V0.reshape((S+1,N,M))
                V0_nonmr.close()

        return loaded_policy, loaded_V0
    return None



coord = lambda x,y: (int((x - storage_grid[0])/step_storage) * step_storage + storage_grid[0], int((y - price_grid[0])/step_price) * step_price + price_grid[0])

coord_array = lambda x,y: (((x - storage_grid[0])/step_storage + 0.5).astype(int), ((y - price_grid[0])/step_price + 0.5).astype(int))

coord_array_exact = lambda x,y: ((x - storage_grid[0])/step_storage, (y - price_grid[0])/step_price)

coord_array_upper = lambda x,y,x_cond,y_cond: (((x - storage_grid[0])/step_storage + 1*(~x_cond)).astype(int), ((y - price_grid[0])/step_price + 1*(~y_cond)).astype(int))

coord_array_lower = lambda x,y: (((x - storage_grid[0])/step_storage).astype(int), ((y - price_grid[0])/step_price ).astype(int))


@jit
def pi_fun_numba(storage, inflow, t, P, q):
    inflow  = coefs_inflow[0] + coefs_inflow[1]* inflow + coefs_inflow[2] * (np.sin((t+coefs_inflow[3])*2*np.pi/12))
    evapor  = coefs_evaporation[0] + coefs_evaporation[1] * np.sin((coefs_evaporation[2]+t)*2*np.pi/12)
    expected_storage = storage + 2.592 * inflow - evapor -  2.592 * q
    return eta * q * ((z_numba(storage, elev_stor) + z_numba(expected_storage,elev_stor))/2 - zd) * P

@jit
def z_numba(I,elev_stor):
    if I <= elev_stor[0,1]:
        return elev_stor[0,0]

    if I >= elev_stor[-1,1]:
        return elev_stor[-1,0]

    for idx in range(1,len(elev_stor)):
        prev_storage = elev_stor[idx-1][1]
        curr_storage = elev_stor[idx][1]

        if prev_storage == curr_storage:
            continue

        if I >= prev_storage and I < curr_storage:
            prev_height = elev_stor[idx-1][0]
            curr_height = elev_stor[idx][0]

            return prev_height   + (curr_height-prev_height) * (I - prev_storage)/(curr_storage-prev_storage)


@jit
def meshgrid_numba(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j,k] = x[k]  # change to x[k] if indexing xy
            yy[j,k] = y[j]
    return xx, yy

@jit
def coord_array_numba(x,y):
    return (((x - storage_grid[0])/step_storage + 0.5).astype(np.int_), ((y - price_grid[0])/step_price + 0.5).astype(np.int_))

@jit
def coord_array_exact_numba(x,y):
    return ((x - storage_grid[0])/step_storage, (y - price_grid[0])/step_price)

@jit
def coord_array_upper_numba(x,y,x_cond,y_cond):
    return (((x - storage_grid[0])/step_storage + 1*(~x_cond)).astype(np.int_), ((y - price_grid[0])/step_price + 1*(~y_cond)).astype(np.int_))

@jit
def coord_array_lower_numba(x,y):
    return (((x - storage_grid[0])/step_storage).astype(np.int_), ((y - price_grid[0])/step_price ).astype(np.int_))

@jit
def prepro_coord_array_numba(x,y):

    cond_storage = np.logical_or(x<storage_grid[0], x>storage_grid[-1])
    x[x<storage_grid[0]]  = storage_grid[0]
    x[x>storage_grid[-1]] = storage_grid[-1]

    cond_price = np.logical_or(y<price_grid[0], y>price_grid[-1])
    y[y<price_grid[0] ] = price_grid[0]
    y[y>price_grid[-1]] = price_grid[-1]
    #cond_price = np.logical_or(x<price_grid[0], x>price_grid[-1])

    return x,y, cond_storage, cond_price

@jit
def next_storage_numba(prev_inflow, t, prev_storage, outflow, shock_n):
    inflow  = coefs_inflow[0] + coefs_inflow[1]* prev_inflow + coefs_inflow[2] * (sin((t+coefs_inflow[3])*2*pi/12))
    evapor  = coefs_evaporation[0] + coefs_evaporation[1] * sin((coefs_evaporation[2]+t)*2*pi/12)
    storage = prev_storage + 2.592 * inflow - evapor -  2.592 * outflow
    return storage + storage_sd * shock_n


@jit
def gridding(future_storage, future_price):
    future_storage,         future_price, cond_storage, cond_price = prepro_coord_array_numba(future_storage, future_price)
    future_storage_upper,   future_price_upper                     = coord_array_upper_numba(future_storage, future_price,cond_storage, cond_price) # at this point future_price is the coord in V
    future_storage_lower,   future_price_lower                     = coord_array_lower_numba(future_storage, future_price)
    future_storage_closest, future_price_closest                   = coord_array_numba(future_storage, future_price) # at this point future_price is the coord in V

    #print(future_price_closest)

    future_storage_upper[cond_storage] = future_storage_closest[cond_storage]
    future_price_upper[cond_price] = future_price_closest[cond_price]
    future_storage_exact, future_price_exact = coord_array_exact_numba(future_storage, future_price)

    weight_storage = np.ones(K) - future_storage_exact + future_storage_lower
    weight_price   = np.ones(J) - future_price_exact   + future_price_lower

    W_storage = np.outer(weight_storage, np.ones(K))
    W_price   = np.outer(weight_price,   np.ones(J))
    W = (W_storage + W_price)/2

    future_storage_lower_cd ,future_price_lower_cd = meshgrid_numba(future_storage_lower, future_price_lower)
    future_storage_upper_cd ,future_price_upper_cd = meshgrid_numba(future_storage_upper, future_price_upper)

    return future_storage_lower_cd, future_price_lower_cd, future_storage_upper_cd, future_price_upper_cd, W

@jit
def model_prices_2_rever_numba_sensib(t,prev, scale = 1):
    return prev + coefs_prices_rever_2[0] * scale +  coefs_prices_rever_2[1] * prev + coefs_prices_rever_2[2] * (np.sin((t+coefs_prices_rever_2[3])*2*np.pi/6)) + coefs_prices_rever_2[4] * (np.sin((t+coefs_prices_rever_2[5])*2*np.pi/12))

@jit
def next_price_2_rever_numba_sensib(t, prev, shock_m, scale = 1):
    return model_prices_2_rever_numba_sensib(t, prev, scale)  + prices_2_rever_sd * shock_m


@jit
def future_payments_numba(V,A,shocks_storage, shocks_price, inflow, current_time,
                          storage, q, price, price_grid, storage_grid,boundaries, scale = 1):

    #future_payment = np.zeros((K,J))

    future_storage = next_storage_numba(inflow, current_time, storage, q, shocks_storage)
    cond = np.logical_or(future_storage < boundaries[0], future_storage > boundaries[1])

    ###################

    future_price = next_price_2_rever_numba_sensib(current_time, price  , shocks_price, scale) # adjust the price model

    ###################

    cond = meshgrid_numba(cond,future_price)[0]

    under_storage = False
    over_storage  = False

    if future_storage.min() < boundaries[0]:
        under_storage = True
        over_storage  = False

    elif future_storage.max() > boundaries[1]:
        under_storage = False
        over_storage  = True


    future_storage_lower_cd, future_price_lower_cd, future_storage_upper_cd, future_price_upper_cd, W = gridding(future_storage, future_price)

    future_value = np.zeros((K,J))
    for idx_storage,_ in enumerate(future_storage_lower_cd[0,:]):
        for idx_price,_ in enumerate(future_price_lower_cd[:,0]):
            future_value[idx_storage,idx_price] = (W[idx_storage, idx_price] * V[future_storage_lower_cd[idx_storage,idx_price],future_price_lower_cd[idx_storage,idx_price] ] +
            (1-W[idx_storage, idx_price]) * V[future_storage_upper_cd[idx_storage,idx_price],future_price_upper_cd[idx_storage,idx_price]])

    future_payment = np.multiply(future_value, A)

    sol = 0
    for x in range(cond.shape[0]):
        for y in range(cond.shape[1]):
             if np.logical_not(cond[x,y]):
                sol += future_payment[x,y]

    return sol, under_storage, over_storage

@jit
def action_value_numba(s, inflow, current_time, storage, q, price, V0, A, shocks_storage,
                                                shocks_prices, price_grid, storage_grid, boundaries, scale):

    s_next = s+1
    next_state_value, under_storage, over_storage = future_payments_numba(V0[s_next],A, shocks_storage, shocks_prices,
                                                            inflow, current_time, storage, q, price, price_grid,
                                                                    storage_grid, boundaries, scale)

    next_state_value = discount * next_state_value

    if under_storage or over_storage:
        return 0 + next_state_value
    else:
        return pi_fun_numba(storage, inflow, current_time, price, q) + next_state_value

@jit
def iv_algo_price(policy, V0, V1, scale):
    for s in range(S):
        current_time = time_0 + s
        inflow       = inflow_values[current_time]

        for n in range(N):
            storage      = storage_grid[n]

            for m in range(M):
                price = price_grid[m]
                rewards = np.zeros(L+1)
                i = 0
                #input_next_storage = [model_inflow, inflow , model_evaporation, current_time,
                #          storage, 0, model_storage]

                input_next_price   = np.array([current_time, price])

                for q in outflow_grid:

                    rewards[i] = (action_value_numba(s, inflow, current_time,
                                                storage, q, price, V0, A, shocks_storage,
                                                shocks_prices, price_grid, storage_grid, boundaries, scale))
                    i+=1

                policy[s,n,m] = outflow_grid[np.argmax(rewards)]
                V1[s,n,m]     = np.max(rewards)

    return policy, V1

prices_data = pd.read_csv('prices_processed.csv', index_col=0)
start = np.where(prices_data.index == 'Dec 2003')[0][0]
end   = np.where(prices_data.index == 'Jan 2020')[0][0]
new_prices = prices_data.iloc[start:end, -1].values

ts = pd.read_csv('Preprocessed_data/ts.csv')
ts.loc[:,'price'] = new_prices

log_diff_prices = ts.price.values
log_diff_prices = np.array(log_diff_prices, dtype=float)
log_diff_prices = np.diff(np.log(log_diff_prices))
log_diff_prices = np.concatenate((np.array([0]), log_diff_prices))
ts['log_diff_prices'] = log_diff_prices

yearmonth = list(ts['yearmonth'])
x_label   = data_tools.from_index_to_dates(yearmonth)

ts['X']                = np.zeros(ts.shape[0])
ts['price_residual']   = np.zeros(ts.shape[0])
ts['storage_residual'] = np.zeros(ts.shape[0])

for t in ts.index:
    if t == 0:
        continue
    #storage residuals

    pred_storage = model_storage(model_inflow, ts.loc[t-1,'inflow'] , model_evaporation, t, ts.loc[t-1,'storage'], ts.loc[t,'outflow'])

    ts.loc[t,'storage_residual'] = ts.loc[t,'storage'] - pred_storage

    #prices residuals

    curr_price = ts.loc[t  , 'price']
    prev_price = ts.loc[t-1, 'price']

    if t < 12:
        ts.loc[t,'X'] = residual(prev_price,curr_price,t)
        ts.loc[t,'price_residual'] = 0
        continue


    prev_x   = ts.loc[t-1 ,'X']
    prev_y_x = ts.loc[t-12,'X']

    ts.loc[t,'X'] = residual(prev_price, curr_price, t)

    expected_X = mean_rever(prev_x,prev_y_x)

    ts.loc[t,'price_residual'] = residual_with_X(prev_price,curr_price,expected_X,t)


# correlation between residuals:

storage_res = ts['storage_residual'].values
prices_res  = ts['price_residual'  ].values

correlation = np.corrcoef(ts['storage_residual'].values[12:], ts['price_residual'].values[12:])[0,1]

eta = 0.5
zd  = 105.69451612903225
r   = 0.0041

discount = 1/(1+r)

elev_stor = zip(ts.elevation.values,ts.storage.values)
elev_stor = sorted(elev_stor, key=lambda x: (x[1],x[0]))



# N = 40 # For the storage grid
# M = 80 # For the price grid
# L = 20 # For the outflow grid
# S = 30 # For the states
N = 40 # For the storage grid
M = 120 # For the price grid
L = 20 # For the outflow grid
S = 80 # For the states

discount = 1/(1+r)

P_min = 3
P_max = 19

I_min = 1750
I_max = 3500
boundaries = np.array([2000,3000])

q_min = 0
q_max = 245

price_grid   = get_middle_points(np.arange(P_min, P_max+ 2**-15, (P_max - P_min)/M))
storage_grid = get_middle_points(np.arange(I_min, I_max+ 2**-15, (I_max - I_min)/N))


rho = 2.33
# K = 30 # number of shocks in the storage grid
# J = 30 # number of shocks in the prices grid

K = 10 # number of shocks in the storage grid
J = 10 # number of shocks in the prices grid


shocks_storage  = get_middle_points(np.arange(-rho, rho+2**-15, (2*rho)/ K))

dist_shocks_storage = (shocks_storage[1] - shocks_storage[0])/2
prob_shocks_storage = norm.cdf(shocks_storage[:] + dist_shocks_storage) - norm.cdf(shocks_storage[:] - dist_shocks_storage)

shocks_prices  = get_middle_points(np.arange(-rho, rho+2**-15, (2*rho)/ J))

dist_shocks_prices = (shocks_prices[1] - shocks_prices[0])/2
prob_shocks_prices = norm.cdf(shocks_prices[:] + dist_shocks_prices) - norm.cdf(shocks_prices[:] - dist_shocks_prices)



outflow_grid = np.arange(q_min, q_max+ 2**-15, (q_max - q_min)/L) # L+1 elements

step_price   = price_grid[1]   - price_grid[0]
step_storage = storage_grid[1] - storage_grid[0]

time_0 = ts.shape[0]
ts_forecast = ts.copy()

for i in range(S):

    current_time = time_0 + i

    prev_inflow  = ts_forecast.loc[current_time - 1, 'inflow' ]
    prev_storage = ts_forecast.loc[current_time - 1, 'storage']

    current_storage = model_storage(model_inflow,prev_inflow, model_evaporation, current_time, prev_storage, prev_inflow)
    current_inflow  = model_inflow(prev_inflow, current_time)

    ts_forecast.loc[current_time,'storage'] = current_storage
    ts_forecast.loc[current_time,'inflow']  = current_inflow


inflow_values = ts_forecast.inflow.values

coefs_inflow = loadtxt('Models/coeficients/inflow_coef.csv')
coefs_evaporation = loadtxt('Models/coeficients/evaporation.csv')

elev_stor = np.array(elev_stor)

def saving_scale_results(policy = None, V0 = None, scale = None):
    policy.tofile(f'Results/policy_2_mr_model_{scale}.txt')
    V0.tofile(f'Results/V0_2_mr_model_{scale}.txt')

def load_scale_results(scale):
    mr_file = open(f'Results/policy_2_mr_model_{scale}.txt')
    loaded_policy = np.fromfile(mr_file)
    loaded_policy = loaded_policy.reshape((S,N,M))
    mr_file.close()

    V0_mr = open(f'Results/V0_2_mr_model_{scale}.txt')
    loaded_V0 = np.fromfile(V0_mr)
    loaded_V0 = loaded_V0.reshape((S+1,N,M))
    V0_mr.close()

    return loaded_policy, loaded_V0



def simulating_paths(loaded_policy, scale,T = 10000, t0 = time_0, storage_0 = ts_forecast.loc[time_0 - 1, 'storage'], price_0 = ts_forecast.loc[time_0 - 1, 'price'], inflow_0 = ts_forecast.loc[time_0 - 1, 'inflow']):
    z_s = np.random.randn(T,S)
    z_p_hat = np.random.randn(T,S)
    z_p = correlation * z_s + np.sqrt(1-correlation**2) * z_p_hat

    storage = np.zeros((T,S+1))
    storage[:,0] = storage_0

    inflow = np.zeros((T,S+1))
    inflow[:,0] = inflow_0

    price = np.zeros((T,S+1))
    price[:,0] = price_0

    outflows_grid = np.zeros((T,S))

    previous_residuals_storage = np.zeros((T,S))

    for s in range(S):
        current_time = time_0 + s

        if s == 0:
            prev_x   = np.zeros(T) + ts_forecast.loc[time_0-1 , 'X']
            prev_y_x = np.zeros(T) + ts_forecast.loc[time_0-12, 'X']

        elif s < 12:
            prev_x   = previous_residuals_storage[:,s]
            prev_y_x = np.zeros(T) + ts_forecast.loc[current_time - 12, 'X']

        else:
            prev_x   = previous_residuals_storage[:, s-1]
            prev_y_x = previous_residuals_storage[:, s-12]


        storage_coord, price_coord, _, _ = prepro_coord_array(storage[:,s], price[:,s])
        storage_coord, price_coord = coord_array(storage_coord, price_coord) # at this point future_price is the coord in V


        outflows = loaded_policy[s,:,:][storage_coord, price_coord]
        outflows_grid[:,s] = outflows

        #print(outflows)

        input_next_storage = [model_inflow, ts_forecast.loc[current_time - 1,'inflow'] , model_evaporation, current_time,
                                              storage[:,s], outflows, model_storage] # Outflow changed from IV loop


        storage[:,s+1] = next_storage(*input_next_storage, z_s[:,s])
        inflow[:, s+1] = model_inflow(inflow[:,s], current_time)
        price[:,s+1] = next_price_2_rever_numba_sensib(current_time, price[:,s], z_p[:,s], scale) # adjust the price model

        previous_residuals_storage[:, s] = 0 # np.log(price[:,s+1] / next_price_mr(t,price[:,s], 0))

    return storage, price, outflows_grid, inflow


def plot_conf_int_boot(sample,scale, quantiles = [0.05, 0.95], var_name = 'noname'):

    T = sample.shape[0]
    S = sample.shape[1]

    i_5  = int(T*quantiles[0])
    i_95 = int(T*quantiles[1])
    i_50 = int(T*0.5)

    top = np.zeros(S-10)
    med = np.zeros(S-10)
    bot = np.zeros(S-10)

    for t in range(S-10):

        sample_t = sorted(sample[:,t])

        top[t] = sample_t[i_95]
        med[t] = sample_t[i_50]
        bot[t] = sample_t[i_5]

    x = range(S-10)
    plt.plot(x, med , label = f'Scaled by {scale}')
    #ax.fill_between(x, bot, top, where=top >= bot, facecolor='green', interpolate=True, alpha = 0.6, label= '5% - 95% confidence interval')
    plt.legend()
    #plt.title(f'Forward simulation: {T} trajectories ',fontdict={'size':'18'})
    plt.ylabel(var_name,fontdict={'size':'16'})
    plt.xlabel('Months',fontdict={'size':'16'})
    # plt.grid()
    #plt.tight_layout()

    return None

# N = 40 # For the storage grid
# M = 80 # For the price grid
# L = 20 # For the outflow grid
# S = 30 # For the states
N = 40 # For the storage grid
M = 120 # For the price grid
L = 20 # For the outflow grid
S = 80 # For the states
scales     = [0.5, 1, 1.5, 2]
names      = ['sensib_price_I', 'sensib_price_q', 'sensib_price_p']
var_labels = {names[0]: 'Inventory (MCM)', names[1]: 'Outflow ($m^3$/s)', names[2]: 'Electricity price (\xa2/kwh)'}

for name in names:
    for scale in scales:
        results = {}
        loaded_policy, V0 = load_scale_results(scale)
        storage_sim, price_sim, outflows_sim, inflow = simulating_paths(loaded_policy, scale)
        results[names[0]] = storage_sim
        results[names[1]] = outflows_sim
        results[names[2]] = price_sim

        plot_conf_int_boot(results[name],scale, var_name = var_labels[name])

    plt.grid()
    plt.tight_layout()
    plt.savefig(f'Figures/paper/{name}.pdf')
    plt.show()
