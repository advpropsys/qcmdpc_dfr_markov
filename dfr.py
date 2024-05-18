import jax.numpy as jnp
import numpy as np
from scipy.stats import hypergeom, binom
import math
import random

def convertBinary(v):
    return jnp.divide(v,jnp.ones_like(v)*2)

def genRandomVector(k, t):
    randomVector = jnp.zeros(k, dtype = jnp.int32)
    randomPositions = jnp.random.choice(k, t, replace=False)
    for j in randomPositions:
        randomVector[j] = 1
    return randomVector

def encryptMcEliece(G, m, e):
    ciphertext = jnp.copy(jnp.add(jnp.matmul(m, G), e))
    ciphertext = convertBinary(ciphertext)
    return ciphertext

def decryptSuccess(plaintext, decryptedText):
    status = jnp.array_equal(plaintext, decryptedText)
    if (status is True):
        print("Decryption success!")
    else:
        print("Decryption failure!")
    return status

def BF(H, y, N):
    r, n = H.shape
    d = sum(H[0,:] == 1) // 2
    i = 0
    s = convertBinary(jnp.matmul(y, jnp.transpose(H)))
    while ( (sum(s==1) != 0) and i < N ):
        s = convertBinary(jnp.matmul(y, jnp.transpose(H)))
        for j in range(n):
            sigma_j = jnp.matmul(s, H[:,j])
            if (sigma_j >= 0.5 * d):
                y[j] = (1 - y[j]) % 2
        i += 1
    s = convertBinary(jnp.matmul(y, jnp.transpose(H)))
    if (sum(s==1) == 0):
        return y
    else:
        return 0

def XBar(S,t,n,w):
    numer = 0
    denom = 0
    for i in range(1, t+1, 2):
        rho = rhoL(n, w, t, i)
        if ~jnp.isnan(rho) and ~jnp.isinf(rho):
            numer += (i - 1) * rho
            denom += rho
    return S * numer / denom

def rhoL(n, w, t, ell):
    return hypergeom.pmf(ell, n, w, t)

def counterDist(S, XBar, n, w, d, t):
    pi1prime = (S + XBar) / (d * t)
    pi0prime = ((w - 1) * S - XBar) / (d * (n - t))
    return pi1prime, pi0prime

def threshold(d, pi1prime, pi0prime, n, t):
    if (pi1prime > 1):
        pi1prime = 1
    if (pi1prime < 0 or pi0prime > 1 or pi0prime < 0):
        print("pi1prime: %f, pi0prime: %f" % (pi1prime, pi0prime))
        return 'F'
    if (pi1prime == 1):
        if ( t >= (n-t) * (pi0prime ** d) ):
            return d
        else:
            return 'F'
    numer = math.log((n - t) / t) + d * math.log((1 - pi0prime) / (1 - pi1prime))
    denom = math.log(pi1prime / pi0prime) + math.log((1 - pi0prime) / (1 - pi1prime))
    return math.ceil(numer / denom)

def sampling(H, s, T, method):
    r, n = H.shape
    if (method == 1):
        for q in range(n):
            unsatEqn = [idx for idx, s_j in enumerate(s) if s_j == 1]
            i = random.choice(unsatEqn)
            ones = [bit for bit, h_ij in enumerate(H[i,:]) if h_ij == 1]
            j = random.choice(ones)
            if ( sum((s + H[:, j]) == 2) >= T ):
                return j
        return 'F'
    if (method == 2):
        sigmaJ = jnp.matmul(s, H)
        toFlip = []
        for k in range(n):
            if (sigmaJ[k] >= T):
                toFlip += [k]
        if (len(toFlip) > 0):
            return random.choice(toFlip)
        else:
            return 'F'
    if (method == 3):
        sigmaJ = jnp.matmul(s, H)
        j = jnp.argmax(sigmaJ)
        if (sigmaJ[j] < T):
            return 'F'
        else:
            maxIndices = []
            for i in range(len(sigmaJ)):
                if (sigmaJ[i] == sigmaJ[j]):
                    maxIndices += [i]
            return random.choice(maxIndices)

def SBSBF(H, y, w, t, N, codeword, samp_method):
    r, n = H.shape
    d = w // 2
    iteration = 1
    flipped = 1
    s = convertBinary(jnp.matmul(y, jnp.transpose(H)))
    while (jnp.count_nonzero(s) > 0 and iteration <= N and flipped == 1):
        flipped = 0
        iteration += 1
        S = sum(s==1)
        t = sum(convertBinary(jnp.array(codeword) + jnp.array(y)) == 1)
        X_bar = XBar(S,t,n,w)
        pi1prime, pi0prime = counterDist(S, X_bar, n, w, d, t)
        T = threshold(d, pi1prime, pi0prime, n, t)
        T = max(math.floor(d / 2) + 1, T)
        j = sampling(H, s, T, samp_method)
        if (j == 'F'):
            print("Cannot sample a bit")
            return 0
        else:
            y[j] = y[j] ^ 1
            flipped = 1
        s = jnp.matmul(y, jnp.transpose(H))
        s = convertBinary(s)
    print("Decrypted text:\n", y)
    print("Codeword:\n", codeword)
    if (sum(s == 1) == 0):
        return y
    else:
        print("Cannot decode")
        return 0


def rhobar(n, w, t):
    temp = 0
    for ell in range(1,t+1,2):
        temp += rhoL(n,w,t,ell)
    return temp

def pL(d, pi1prime, pi0prime, p, T, t, n):
    prob1 = 0
    prob2 = 0
    for i in range(T):
        prob1 += binom.pmf(i, d, pi1prime)
        prob2 += binom.pmf(i, d, pi0prime)
    pL = (prob1 ** t) * (prob2 ** (n - t))
    return pL

def calcP(T, n, d, t, w, S, pi1prime, pi0prime):
    p = 0
    for i in range(T):
        sigma = i
        p_sigma_neg = t * sigma * binom.pmf(sigma, d, pi1prime) / (w * S)
        p_sigma_pos = (n - t) * sigma * binom.pmf(sigma, d, pi0prime) / (w * S)
        p += p_sigma_neg + p_sigma_pos
    return p

def calcQ(T, n, d, t, w, S, pi1prime, pi0prime):
    q = 0
    for i in range(T):
        sigma = i
        q_sigma_neg = t * binom.pmf(sigma, d, pi1prime) / n
        q_sigma_pos = (n - t) * binom.pmf(sigma, d, pi0prime) / n
        q += q_sigma_neg + q_sigma_pos
    return q

def q_maxs(n, d, t, pi1prime, pi0prime, sigma):
    temp1 = 0
    temp0 = 0
    for x in range(sigma):
        temp1 += binom.pmf(x, d, pi1prime)
        temp0 += binom.pmf(x, d, pi0prime)
    temp1_new = temp1 + binom.pmf(sigma, d, pi1prime)
    temp0_new = temp0 + binom.pmf(sigma, d, pi0prime)
    prob_max = (temp1_new ** t) * (temp0_new ** (n-t)) - (temp1 ** t) * (temp0 ** (n-t))
    denom = t * binom.pmf(sigma, d, pi1prime) + (n - t) * binom.pmf(sigma, d, pi0prime)
    q_max_minus = t * binom.pmf(sigma, d, pi1prime) / denom * prob_max
    q_max_plus = (n - t) * binom.pmf(sigma, d, pi0prime) / denom * prob_max
    return q_max_plus, q_max_minus

def DFR_model(t_pass, t_fail, n, w, t_init, prob, samp_method):
    d = w // 2
    r = n // 2
    maxS = r
    DFR = {}
    while (prob[maxS] < 10 ** (-30)):
        maxS -= 1
    if ( (maxS + t_init) % 2 == 1 ):
        maxS -= 1
    print("maxS:", maxS)
    minS = int(math.floor(d / 2)) + 1
    print("minS:", minS)
    for t in range(0, t_fail + 1):
        DFR[(0,t)] = 0
    for S in range(1, minS):
        DFR[(S,0)] = 0
        for t in range(1, t_fail):
            DFR[(S,t)] = 1
    for S in range(1, r + 1):
        DFR[(S, t_fail)] = 1
    for S in range(minS, r + 1):
        DFR[(S, t_pass)] = 0
    for S in range(minS, maxS + 1):
        for t in range(t_pass + 1, t_fail):

            X_bar = XBar(S,t,n,w)
            pi1prime, pi0prime = counterDist(S, X_bar, n, w, d, t)
            T = threshold(d, pi1prime, pi0prime, n, t)
            if (T == 'F'):
                T = int(minS)
            T = max(minS, T)
            print("T:", T)
            p = calcP(T, n, d, t, w, S, pi1prime, pi0prime)
            PL = pL(d, pi1prime, pi0prime, p, T, t, n)
            q = calcQ(T, n, d, t, w, S, pi1prime, pi0prime)
            if (samp_method == 1):
                DFR[(S,t)] = PL
            elif (samp_method == 2):
                DFR[(S,t)] = PL
            elif (samp_method == 3):
                DFR[(S,t)] = pL(d, pi1prime, pi0prime, p, minS, t, n)
            if (samp_method == 1):
                for sigma in range(T, min(d + 1, S + 1)):
                    print("(S,t,sigma) = (%d,%d,%d)" % (S,t,sigma))
                    p_sigma_neg = t * sigma * binom.pmf(sigma, d, pi1prime) / (w * S)
                    p_sigma_pos = (n - t) * sigma * binom.pmf(sigma, d, pi0prime) / (w * S)
                    p_sigma_neg_prime = p_sigma_neg * (1 - PL) / (1 - p)
                    p_sigma_pos_prime = p_sigma_pos * (1 - PL) / (1 - p)
                    DFR[(S,t)] += p_sigma_neg_prime * DFR[S + d - 2 * sigma, t - 1] + p_sigma_pos_prime * DFR[S + d - 2 * sigma, t + 1]
            elif (samp_method == 2):
                for sigma in range(T, min(d + 1, S + 1)):
                    print("(S,t,sigma) = (%d,%d,%d)" % (S,t,sigma))
                    q_sigma_neg = t * binom.pmf(sigma, d, pi1prime) / n
                    q_sigma_pos = (n - t) * binom.pmf(sigma, d, pi0prime) / n
                    q_sigma_neg_prime = q_sigma_neg * (1 - PL) / (1 - q)
                    q_sigma_pos_prime = q_sigma_pos * (1 - PL) / (1 - q)
                    DFR[(S,t)] += q_sigma_neg_prime * DFR[S + d - 2 * sigma, t - 1] + q_sigma_pos_prime * DFR[S + d - 2 * sigma, t + 1]
            elif (samp_method == 3):
               for sigma in range(minS, min(d + 1, S + 1)):
                  #  print(min(d + 1, S + 1))
                   print("(S,t,sigma) = (%d,%d,%d)" % (S,t,sigma))
                   q_max_plus, q_max_minus = q_maxs(n, d, t, pi1prime, pi0prime, sigma)
                   DFR[(S,t)] += q_max_minus * DFR[S + d - 2 * sigma, t - 1] + q_max_plus * DFR[S + d - 2 * sigma, t + 1]
    fail = 0

    if (t_init % 2 == 0):
        startS = 2
    else:
        startS = 1


    for S in range(startS, maxS + 1, 2):
        if (S,t_init) in DFR:
          if (math.isnan(DFR[(S,t_init)]) or math.isnan(prob[S])):
              continue
          fail += prob[S] * DFR[(S,t_init)]

    return(fail)

def g_0(n, w, t, rhobar, k):
    if (k % 2 == 1):
        return 0
    else:
        return rhoL(n, w, t, k) / (1- rhobar)

def g_1(n, w, t, rhobar, k):
    if (k % 2 == 0):
        return 0
    else:
        return rhoL(n, w, t, k) / rhobar

def convolveH(n, w, t, rhobar):
    d = w // 2
    r = n // 2

    G0 = [g_0(n, w, t, rhobar, i) for i in range(w+1)]
    G1 = [g_1(n, w, t, rhobar, i) for i in range(w+1)]

    G0conv = [[1]]
    G1conv = [[1]]

    for i in range(r):
        G0conv += [np.convolve(np.asarray(G0conv[i]), np.asarray(G0))]
        G1conv += [np.convolve(np.asarray(G1conv[i]), np.asarray(G1))]

    h = [G0conv[r][d * t]]

    # compute h(ell) for ell = 1, ..., r-1
    for i in range(1, r):
        temp = 0
        j = 0
        while (j >= 0 and j < len(G1conv[i]) and d * t - j >=0 and
               d * t - j < len(G0conv[r - i])):
            temp += G1conv[i][j] * G0conv[r - i][d * t - j]
            j += 1
        h += [temp]

    h += [G1conv[r][d * t]]

    return h

def syndrome_distance(n, w, t, rhobar):
    r = n // 2
    d = w // 2

    if (t == 1):
        return [0 for _ in range(d)] + [1] + [0 for _ in range(r - d)]

    h = convolveH(n, w, t, rhobar)
    sDist = []

    for ell in range(r+1):
        denom = 0
        for k in range(r):
            denom += binom.pmf(k, r, rhobar) * h[k]
        sDist += [binom.pmf(ell, r, rhobar) * h[ell] / denom]

    return sDist

# test parameters

n0 = 2
r = 750
wi = 39

t = 19

samp_method = 3
n = n0 * r
w = wi * n0
t_pass = 5
t_fail = 23


rbar = rhobar(n,w,t)
prob = syndrome_distance(n,w,t, rbar)

print("\n\nMC DFR =", DFR_model(t_pass, t_fail, n, w, t, prob, samp_method))