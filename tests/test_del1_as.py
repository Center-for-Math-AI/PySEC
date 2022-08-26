from PySEC import del0, del1, generate_data
import os, torch


def test_del1_as_stub(): # stub test for calling and return signature
    num_points = 360
    n0 = 200
    n1 = 20
    data, iparams = generate_data.generate_circle(num_points)
    u0, l0, d0 = del0.del0(data.T, n0)
    try:
        U, L, D1, G, H, cijk = del1.del1_as_einsum(u0, l0, d0, n1)
        assert True
    except Exception:
        assert False
