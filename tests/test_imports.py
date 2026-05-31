def test_public_imports():
    from sc_ipca import IPCA, Instruments, ipca

    assert IPCA is ipca
    assert Instruments.__name__ == "Instruments"
