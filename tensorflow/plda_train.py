import sys

def EM(speaker_observations, F, D):
    fi = {}
    S = 0
    Li_inv = {}
    yi_hat = {}
    R = 0
    T = 0

    objective = 0

    FTDF = np.linalg.multi_dot([F.T, D, F])
    N = 0
    for spk, obs in speaker_observations.items():
        N += len(obs)
        fi[spk] = np.sum(obs)
        S += np.sum([np.outer(o, o) for o in obs])
        Li_inv[spk] = np.linalg.inv(len(obs) * FTDF + 1)
        yi_hat[spk] = np.linalg.multi_dot[Li_inv[spk], F.T, D, fi[spk]]
        R += len(obs) * (Li_inv[spk] + np.outer(yi_hat, yi_hat))
        T += np.outer(yi_hat[spk], fi[spk])

    R_inv = np.linalg.inv(R)
    F_new = np.dot(R_inv, T).T
    D_new = np.linalg.inv((S - np.linalg.multi_dot(T.T, R_inv, T)) / N)

    return F_new, D_new

if __name__ == '__main__':
    
