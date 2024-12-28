""" Simple Arthimetic unit testing"""


def test_simple_multiplication():
    """ Simple assertion: check if 2 * 8 equals 16 """
    assert 2 * 8 == 16


def test_addition():
    """ Another simple test for addition """
    assert 3 + 4 == 7


if __name__ == "__main__":
    test_simple_multiplication()
    test_addition()
    print("All tests passed!")
