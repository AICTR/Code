import numpy as np
import pandas as pd
from trend_represntations import trend_representations
from simplex_projection import simplex_projection_selfnorm2

def radial_basis_function(close_price,data,tplus1,daily_port, datahat,datahat_center,EMA,win_size,
        epsilon=1000, alpha=0.5, sigmasquare=0.0025):
    '''

    :param epsilon: a parameter that controls the update step size
    :param alpha: the mixing parameter of EMA
    :param sigmasquare: the scale parameter of the RBFs
    :return:
    '''

    [T, nstk] = data.shape
    EMA, SMA, PP = trend_representations(close_price,nstk,data,tplus1,EMA,win_size,alpha=0.5)

    xhat = np.zeros((nstk, 3))
    xhat[:, 0]=EMA.transpose().reshape(nstk)
    xhat[:, 1]=SMA.transpose().reshape(nstk)
    xhat[:, 2]=PP.transpose().reshape(nstk)

    datahat[tplus1,:,:]=xhat
    xhat_simplex = np.zeros((nstk, 3))
    xhat_simplex[:, 0] = simplex_projection_selfnorm2(xhat[:, 0], 1)
    xhat_simplex[:, 1] = simplex_projection_selfnorm2(xhat[:, 1], 1)
    xhat_simplex[:, 2] = simplex_projection_selfnorm2(xhat[:, 2], 1)

    datahat_center[tplus1,:,:]=xhat_simplex
    if (tplus1 < win_size + 2):
        rate = np.zeros((tplus1-1, 3))
        for id in range(0,3):
            rate[:, id] = np.diag(np.dot(datahat_center[0: tplus1 - 1, :, id], (data[0: tplus1 - 1, :].transpose())))

    else:
        rate = np.zeros((win_size, 3))
        for id in range(0, 3):
            rate[:, id]=np.diag(np.dot(datahat_center[(tplus1 - win_size): (tplus1 - 1),:, id],
                                       (data[(tplus1 - win_size): (tplus1 - 1),:]).transpose()))

    rate_score = rate.min(0).transpose()
    idx = np.argmax(rate_score)

    ones_w = np.ones((1, 3))
    center = xhat_simplex

    bigA = xhat

    eunorm = np.diag(np.dot((np.dot(xhat_simplex[:, idx],ones_w) - center).transpose(),
                     np.dot(xhat_simplex[:,idx],ones_w)-center))
    bigphi = np.exp(-eunorm / (2 * sigmasquare))
    onesn = np.ones((nstk, 1))
    dee = np.dot((np.identity(nstk) - np.dot(onesn,onesn.transpose())/(nstk)),(np.dot(bigA,bigphi)))
    daily_port = daily_port.reshape(nstk) + (epsilon * dee / np.linalg.norm(dee))
    daily_port = simplex_projection_selfnorm2(daily_port, 1)
    return daily_port,datahat,datahat_center

