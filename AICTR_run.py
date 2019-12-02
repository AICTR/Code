import numpy as np
import pandas as pd
from trend_represntations import trend_representations
from simplex_projection import simplex_projection_selfnorm2
from AICTR import radial_basis_function
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

def AICTR_run(data,win_size = 5,tran_cost = 0):
    [T, N] = data.shape
    # create np.array
    EMA = np.ones((1, N))
    cum_wealth = np.ones((T, 1))
    daily_incre_fact = np.ones((T, 1))
    daily_port = np.ones((N, 1)) / N  # first-day, average-wetight
    daily_port_total = np.ones((N, T)) / N
    daily_port_o = np.zeros((N, 1))
    datahat = np.ones((T, N, 3))
    datahat_center = np.ones((T, N, 3)) / N
    close_price = np.ones((T, N))

    for i in range(1, T):
        close_price[i, :] = close_price[i - 1, :] * data[i, :]

    run_ret = 1
    for t in range(0, T):
        daily_port_total[:, t] = daily_port.reshape(N)  # record_daily_weight
        daily_incre_fact[t, 0] = np.dot(data[t, :], daily_port) * (
                    1 - tran_cost / 2 * np.sum(abs(daily_port - daily_port_o)))  # calculate weighted return
        run_ret = run_ret * daily_incre_fact[t, 0]  # accumulative return
        cum_wealth[t] = run_ret  # record daily accumulative return
        daily_port_o = daily_port * ((data[t, :].transpose()).reshape(N, 1)) / (np.dot(data[t, :], daily_port))  # price_weighted
        if (t < T-1):
            daily_port_n, datahat, datahat_center = radial_basis_function(close_price, data, t + 1, daily_port, datahat,
                                                                          datahat_center, EMA, win_size)
    return cum_wealth,daily_incre_fact,daily_port_total

if __name__ == '__main__':
    register_matplotlib_converters()
    hs = pd.read_excel('HS300.xlsx')
    data = hs.copy()
    data.set_index(['Date'], inplace=True)
    data = data.values
    cum_wealth,daily_incre_fact,daily_port_total = AICTR_run(data)
    plt.plot(hs['Date'],cum_wealth)
    plt.show()