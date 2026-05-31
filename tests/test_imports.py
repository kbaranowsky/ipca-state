def test_public_imports():
    from ipca import IPCA, Instruments, ipca

    assert IPCA is ipca
    assert Instruments.__name__ == "Instruments"


def test_package_version():
    import ipca

    assert hasattr(ipca, "__version__")