import statsmodels.api as sm
import pandas as pd
import numpy as np
class linnear_model():
    def __init__(self,data,label,fac_cols):
        self.data = data
        self.label = label
        self.fac_cols = fac_cols
    def regress(self,data):
        Y = data[self.label]
        X = data[self.fac_cols]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        return results.params
    def predict(self,data,params):
        i= 1
        score = 0
        for col in self.fac_cols:
            score = params[i]*data[col]+score
            i += 1
        data['score'] = score
        return data
    def compute_score(self):
        dates = self.data['trade_date'].unique()
        dates.sort()
        param = []

        for date in  dates[:-1]:
            param.append([ 0 if i<=0 else i for  i in self.regress(self.data[self.data['trade_date']==date])])
        print(param)
        param = np.nansum(param,axis=0)
        # param = [1,1,1,1,2,1]
        print(param,'-'*10)
        print(dates[-1])
        pre_data = self.data[self.data['trade_date']==max(dates)]
        res = self.predict(pre_data,param)
        return res