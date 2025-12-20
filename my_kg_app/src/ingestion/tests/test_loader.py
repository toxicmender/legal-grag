from ..loader import load_pdf


def test_load_pdf_returns_string():
    assert isinstance(load_pdf("/dev/null"), str)
