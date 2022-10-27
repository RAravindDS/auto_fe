import pytest 
from AutoFeatureEnginerring import AutoFeatureEngineering
import pandas as pd

data = pd.read_csv(r"train.csv")
out = AutoFeatureEngineering(data).auto_fe()
print(out.isnull().sum())


test_cases = [ (data, pd.DataFrame), (data, pd.DataFrame), (data, pd.DataFrame)]

@pytest.mark.parametrize("data, response", test_cases)

def test_auto_feature_engineering(data, response):
    out = AutoFeatureEngineering(data).auto_fe()
    assert out == response 