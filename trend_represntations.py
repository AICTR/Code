import numpy as np
import pandas as pd

def trend_representations(close_price,nstk,data,tplus1,EMA,win_size,alpha=0.5):
    EMA = alpha + (1 - alpha) * EMA / data[tplus1 - 1, :]
    if (tplus1 < win_size + 2):
        SMA = data[tplus1 - 1, :]
        PP = data[tplus1 - 1, :]
    else:
        SMA = np.zeros(shape=(1, nstk));
        tmp_x = np.ones(shape=(1, nstk));
        for i in range(0, win_size):
            SMA = SMA + 1 / tmp_x
            tmp_x = tmp_x * data[tplus1 - 1 - i, :]
        SMA = SMA * (1 / win_size)

        closebefore = close_price[(tplus1 - win_size):(tplus1 - 1),:]
        closepredict = closebefore.max(0)
        PP = closepredict / close_price[tplus1 - 1,:]
    return EMA, SMA, PP