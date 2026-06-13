# heat_analysis_system/src/models.py
# [Раздел 2.2 Диплома] Кусочно-линейная робастная регрессия Хубера с расчетом Доверительных Интервалов

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as t_student

class RobustHeatModel:
    def __init__(self, config):
        self.cfg = config
        self.beta_0 = None
        self.beta_1 = None
        self.beta_0_ci = (None, None)
        self.beta_1_ci = (None, None)
        self.r2 = None
        self.delta = None

    def fit(self, df):
        t_op = self.cfg['region_settings']['t_op']
        df_heating = df[df['t_out'] <= t_op].copy()
        if len(df_heating) < 10:
            df_heating = df.copy()

        X = df_heating['t_out'].values
        y = df_heating['Q'].values

        X_const = sm.add_constant(X)
        ols_res = sm.OLS(y, X_const).fit()
        ols_residuals = np.abs(ols_res.resid)
        
        q_val = self.cfg['models']['huber']['initial_delta_quantile']
        self.delta = float(np.quantile(ols_residuals, q_val))
        if self.delta < 1e-4: self.delta = 1.345

        huber_model = sm.RLM(y, X_const, M=sm.robust.norms.HuberT(t=self.delta))
        huber_results = huber_model.fit()

        self.beta_0 = float(huber_results.params[0])  
        self.beta_1 = float(huber_results.params[1])  

        alpha = self.cfg['models']['huber']['ci_alpha']
        dof = int(huber_results.df_resid)
        t_crit = t_student.ppf(1 - alpha / 2, dof)
        bse = huber_results.bse  

        self.beta_0_ci = (float(self.beta_0 - t_crit * bse[0]), float(self.beta_0 + t_crit * bse[0]))
        self.beta_1_ci = (float(self.beta_1 - t_crit * bse[1]), float(self.beta_1 + t_crit * bse[1]))

        y_pred_heating = huber_results.predict(X_const)
        w = huber_results.weights
        y_bar_w = np.sum(w * y) / np.sum(w)
        ss_res_w = np.sum(w * (y - y_pred_heating) ** 2)
        ss_tot_w = np.sum(w * (y - y_bar_w) ** 2)
        self.r2 = float(1 - (ss_res_w / max(ss_tot_w, 1e-6)))

        all_X = df['t_out'].values
        preds = []
        for t in all_X:
            if t <= t_op:
                preds.append(self.beta_0 + self.beta_1 * t)
            else:
                preds.append(self.beta_0 + self.beta_1 * t_op)
                
        df['q_pred'] = preds
        df['residual'] = df['Q'] - df['q_pred']
        return df
