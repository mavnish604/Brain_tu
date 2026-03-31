import sys
try:
    print("Loading executorch.runtime...")
    from executorch.runtime import Module
    print("Loaded!")
except Exception as e:
    print(e)
