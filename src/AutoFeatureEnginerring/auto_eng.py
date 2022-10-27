import pandas as pd
import warnings 
from AutoFeatureEnginerring.logger import logger 
warnings.filterwarnings('ignore')


class AutoFeatureEngineering: 
    def __init__(self, xtrain : pd.DataFrame):
        super().__init__()

        """ This helps to impute the missing values automatically
        Examples
        --------
        >>> out = AutoFeatureEngineering(data).autofe()
        >>> out.head()
        
        """ 
        self.data = xtrain 
  
    def int_columns(self):
        """ It will output the columns which is input and float """ 
        try: 
            int_columns = self.data.select_dtypes(exclude = "object")
            return int_columns 
        except Exception as e: 
            logger.error(e)
            raise e
            
        

    def string_columns(self):
        """ It will output the columns which is object""" 
        try: 
            str_columns = self.data.select_dtypes(include = 'object')
            return str_columns 
        except Exception as e: 
            logger.error(e)
            raise e

    def missing_column_finder(self, data : pd.DataFrame): 
        """ It will output the missing columns in data frame """ 
        try: 
            indicators = list(data.isnull().sum())
            missing_columns = [ columns for indicator, columns in zip(indicators, list(data.columns)) if indicator > 0]

            return missing_columns 
        except Exception as e:
            logger.error(e)
            raise e

    def get_missing_values(self): 
        """ It will return the missing columns seperately by obj columns and int columns """ 
        try: 
            str_, int_ = self.string_columns(), self.int_columns()
            missing_values_in_str, missing_values_in_int = self.missing_column_finder(str_), self.missing_column_finder(int_)

            missing_values_in_str = self.data[missing_values_in_str]
            missing_values_in_int = self.data[missing_values_in_int]
            indicators = list(self.data.isnull().sum())
            non_missing_columns = [ columns for indicator, columns in zip(indicators, list(self.data.columns)) if indicator == 0]
            self.non_missing_columns = non_missing_columns 
            non_missing_values_both = self.data[self.non_missing_columns]

            return missing_values_in_str, missing_values_in_int, non_missing_values_both
        except Exception as e: 
            logger.error(e)
            raise e


    def missing_value_imputer_int(self, data:pd.DataFrame, by:str='mean'): 
        """ * This will impute the null values in integer columns. Default it's going to done in mean logic. 
            * If you want to use any advanced technique you can use KNN or MICE 
            
        Parameter
        ---------
        by : str
            mean or KnnImputer or mice or mean
            
        Examples
        --------
        >>> obj.missing_value_imputer_int(by='mean')
        """
        try: 
            col = list(data.columns)
            if by == 'mean':
                from sklearn.impute import SimpleImputer 
                imp = SimpleImputer(strategy="mean")
                
                imp_after = imp.fit_transform(data)
                dataframe = pd.DataFrame(imp_after, columns = col)

                return dataframe 

            if by == 'KnnImputer':
                from sklearn.impute import KNNImputer 
                knn = KNNImputer(n_neighbors = 5, add_indicator = False)
                dataframe = pd.DataFrame(knn.fit_transform(data), columns = col)

                return dataframe

            if by == 'mice':
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer 
                from sklearn.linear_model import LinearRegression 

                lr = LinearRegression()
                imp = IterativeImputer(estimator = lr, verbose = 3, max_iter = 30, tol=1e-10, imputation_order = 'roman')
                dataframe = pd.DataFrame(imp.fit_transform(data), columns = col)
                
                return dataframe

        except Exception as e:
            logger.error(e)
            raise e

  
    def missing_value_imputer_obj(self, data:pd.DataFrame, by:str='frequent'): 
        """ This will impute the categorical data by frequent values or predict those by own model """
        try: 
            col = list(data.columns)

            if by == 'frequent': 
                for co in col: 
                    key = data[co].value_counts().keys()
                    values = list(data[co].value_counts().values).index(data[co].value_counts().values.max())
                    filling_value = key[values]

                    data[co] = data[co].fillna(filling_value)

                return data
        except Exception as e: 
            logger.error(e)
            raise e


    def auto_fe(self, int_imputation_method:str="mean", obj_imputation_method:str='frequent'): 
        try: 
            string_mv, int_mv, non_mv = self.get_missing_values()

            if len(string_mv.values[0]) > 0: 
                string_mv = self.missing_value_imputer_obj(string_mv, obj_imputation_method)

            if len(int_mv.values[0]) > 0: 
                int_mv = self.missing_value_imputer_int(int_mv, int_imputation_method)

            frames = [string_mv, int_mv, non_mv]
            final_df = pd.concat(frames, axis = 1)

            return final_df
        except Exception as e: 
            logger.error(e)
            raise e
