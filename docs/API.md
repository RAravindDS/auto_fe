# API reference 

## One line feature engineering: 
```python 
import pandas as pd 
from AutoFeatureEnginerring import AutoFeatureEngineering
data = pd.read_csv(r"load_csv")

# one line!
out = AutoFeatureEngineering(data).auto_fe()
```

| Args                  | Description                                                                                                  |
|-----------------------|--------------------------------------------------------------------------------------------------------------|
| int_imputation_method | Imputing numerical values. Default method is 'mean'. There are certain methods available! knnImputer or mice | 
| obj_imputation_method | Imputing categorical values. Default method is 'most frequent'                                               |

|Return|Description|
|--|--|
|dataframe|Imputed dataframe|


## Customizable options: 
```python 
import pandas as pd 
from AutoFeatureEnginerring import AutoFeatureEngineering

# get the data 
data = pd.read_csv(r"https://raw.githubusercontent.com/RAravindDS/auto_fe/main/research/train.csv")

# impute with your own algorithm!
out_df = auto_fe_obj.auto_fe(int_imputation_method = 'KnnImputer')  # this will impute the values with KnnImputer 
out_df = auto_fe_obj.auto_fe(int_imputation_method = 'mice') # this will impute the values with mice imputer  with default values!
```


## Bonus Methods!
??? note "example" 
    ```python 
    import pandas as pd 
    from AutoFeatureEnginerring import AutoFeatureEngineering

    # get the data 
    data = pd.read_csv(r"https://raw.githubusercontent.com/RAravindDS/auto_fe/main/research/train.csv")

    # it will grab the numerical columns
    auto_fe_obj.int_columns()  # It will grab the numerical columns! 

    # it will grab the categorical columns
    auto_fe_obj.string_columns()

    # grabbing missing values based on datatype 
    categorical_missing_col, numerical_missing_col, non_missing_col = auto_fe_obj.get_missing_values()   # It will return the dataframe!
    # it contains missing values based on datatypes! 
    ```



