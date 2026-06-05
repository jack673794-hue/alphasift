from alphasift.normalize import normalize_code


def test_normalize_code_accepts_numeric_suffixed_and_prefixed_codes():
    assert normalize_code(1.0) == "000001"
    assert normalize_code("SZ000001") == "000001"
    assert normalize_code("000001.SZ") == "000001"
    assert normalize_code("sh600000") == "600000"
    assert normalize_code("证券代码:300750") == "300750"
