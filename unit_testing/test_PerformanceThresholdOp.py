import pytest
import sys
sys.path.insert(1, 'src/')
print(sys.path)
from PerformanceThresholdOp import PerformanceThresholdOp

# check it returns pass if value higher than threshold
def test_return_pass_if_val_higher_than_threshold():
    assert PerformanceThresholdOp('unit_testing\model_performance_threshold.json', 0.98) == 'pass', "Should be pass"

# check it returns fail if value lower than threshold
def test_return_fail_if_val_lower_than_threshold():
    assert PerformanceThresholdOp('unit_testing\model_performance_threshold.json', 0.9999) == 'fail', "Should be fail"
