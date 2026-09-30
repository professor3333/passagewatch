import passagewatch


def test_version_is_exposed() -> None:
    assert passagewatch.__version__ == "0.1.0"
