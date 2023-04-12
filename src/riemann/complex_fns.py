## Just such a fun one to implement

# # A faster implementation of j for repeated summations
# E4_vals, E6_vals, initialized = [1], [1], False
# def j_fast_init(z, max_precision = 300):
#     q = exp(2 * PI * 1j * z)
#     for n in range(1, max_precision):
#         E4_vals.append(240 * (n ** 3 * q ** n / (1 - q ** n)))
#         E6_vals.append(-504 * (n ** 5 * q ** n / (1 - q ** n)))


# def j_fast(z, precision):
#     q = exp(2 * PI * 1j * z)
#     E4t3, E6t2 = np.ones_like(q), np.ones_like(q)
#     for n in range(1, precision):
#         E4t3 += E4_vals[n]
#         E6t2 += E6_vals[n]
#     E4t3 = E4t3 ** 3
#     E6t2 = E6t2 ** 2
#     return 1728 * E4t3 * inv(E4t3 - E6t2)