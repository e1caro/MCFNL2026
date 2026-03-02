import numpy as np
import matplotlib.pyplot as plt
import pytest

def test_example():
    # Given...
    num1 = 1
    num2 = 1

    # When...
    result = num1 + num2

    # Expect...
    assert result == 2


if __name__ == "__main__":
    pytest.main([__file__])
