import numpy as np

def sparsemax(z):

  """forward pass for sparsemax
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return np.maximum(0, z - tau_z)
  
  def EMMS(Z,Y):

    x = Z
    y = Y
    N, D2, K = y.shape
    for i in range(K):
        y_mean = np.mean(y[:,:,i] , axis=0)
        y_std = np.std(y[:,:,i], axis=0)
        epsilon = 1e-8
        y[:,:,i] = (y[:,:,i] - y_mean) / (y_std + epsilon)
    N,D1 = x.shape
    x_mean = np.mean(x , axis=0)
    x_std = np.std(x, axis=0)
    epsilon = 1e-8
    x = (x - x_mean) / (x_std + epsilon)

    lam = np.array([1/K] * K)
    w1 = 0
    lam1 = 0
    T = 0
    b = np.dot(y, lam)
    T = b
    for k in range(1):
        a = x
        
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        w1 = w
        a = y.reshape(N*D2, K)
        
        b = np.dot(x, w).reshape(N*D2)
        lam = np.linalg.lstsq(a, b, rcond=None)[0]
        lam = lam.reshape(1,K)
        lam = sparsemax(lam)
        lam = lam.reshape(K,1)
        lam1 = lam
        b = np.dot(y, lam)
        b = b.reshape(N,D2)
        T = b
    y_pred = np.dot(x,w1)
    res = np.sum((y_pred - T)**2) / N*D2

    return -res
