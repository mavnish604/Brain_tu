import sys
print('A')
try:
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    print('B')
    m = _load_for_executorch('models/densenet121_brain_tumor.pte')
    print('C')
except Exception as e:
    print('Error:', e)
print('D')
