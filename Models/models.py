
from numpy import loadtxt, sin, pi,log, exp, min, max

coefs_evaporation = loadtxt('Models/coeficients/evaporation.csv')

def model_evaporation(t):
    return coefs_evaporation[0] + coefs_evaporation[1] * sin((coefs_evaporation[2]+t)*2*pi/12)

coefs_inflow = loadtxt('Models/coeficients/inflow_coef.csv')

def model_inflow(t,prev):
    return coefs_inflow[0]*prev+ coefs_inflow[1]*(1-coefs_inflow[0]) + (coefs_inflow[2] * (t% 12 == 1) + coefs_inflow[3] * (t% 12 == 2) + coefs_inflow[4] * (t% 12 == 3) + coefs_inflow[5] * (t% 12 == 4) + coefs_inflow[6] * (t% 12 == 5) + coefs_inflow[7] * (t% 12 == 6) + coefs_inflow[8] * (t% 12 == 7) + coefs_inflow[9] * (t% 12 == 8) + coefs_inflow[10] * (t% 12 == 9) + coefs_inflow[11] * (t% 12 == 10)+ coefs_inflow[12] * (t% 12 == 11))*(1- coefs_inflow[0])


def model_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow):
    inflow      = model_inflow(t,prev_inflow)
    evaporation = model_evaporation(t)

    return prev_storage + 2.592 * inflow - evaporation -  2.592 * outflow

coefs_log_prices = loadtxt('Models/coeficients/log_prices.csv')
coefs_mean_rever = loadtxt('Models/coeficients/mean_rever.csv')

def model_log_prices(t,prev,prev_x,prev_y_x):
    f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
    X = lambda x,x_y: coefs_mean_rever[0]*x + coefs_mean_rever[1] * x_y
    return log(prev) + f(t) + X(prev_x,prev_y_x)

def residual(prev_price, curr_price,t):
    f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
    return log(curr_price) - log(prev_price) - f(t)

storage_sd = 51.67998438309842
prices_sd  = 0.0855231078219911

def next_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow, model_storage, shock_n):
    return model_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow) + storage_sd * shock_n

def next_price(t,prev, shock_m):
    f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
    return prev*exp(f(t) + prices_sd * shock_m)

def z(I,elev_stor):
    unzipped = list(zip(*elev_stor))

    if I <= min(unzipped[1]):
        return elev_stor[0][0]

    if I >= max(unzipped[1]):
        return elev_stor[-1][0]

    for idx in range(1,len(elev_stor)):
        prev_storage = elev_stor[idx-1][1]
        curr_storage = elev_stor[idx][1]

        if prev_storage == curr_storage:
            continue

        if I >= prev_storage and I < curr_storage:
            prev_height = elev_stor[idx-1][0]
            curr_height = elev_stor[idx][0]

            return prev_height   + (curr_height-prev_height) * (I - prev_storage)/(curr_storage-prev_storage)