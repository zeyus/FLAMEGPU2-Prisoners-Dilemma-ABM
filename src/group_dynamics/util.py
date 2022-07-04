# Utility function to get the cuberoot of a number without requiring numpy
def cbrt(x):
    root = abs(x) ** (1/3)
    return root if x >= 0 else -root

