# heat_analysis_system/src/forecast.py
# [Раздел 2.2 Диплома] Модуль краткосрочного предиктивного моделирования нагрузки (ARX)

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

class ShortTermHeatForecaster:
    def __init__(self, cfg):
        self.cfg = cfg
        # Используем робастный регрессор Хубера, чтобы выбросы/сбои в датчиках не ломали прогноз
        self.model = HuberRegressor(epsilon=1.35, max_iter=1000)
        self.is_trained = False

    def prepare_features(self, df_history):
        """
        Формирует лаговые переменные (предысторию) для учета тепловой инерции МКД
        """
        df = df_history.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Лаг расхода: потребление тепла вчера (Q_инерция)
        df['Q_lag1'] = df['Q'].shift(1)
        # Направление изменения температуры: (T_ул_сегодня - T_ул_вчера)
        df['t_out_delta'] = df['t_out'] - df['t_out'].shift(1)
        
        # Удаляем строки с NaN, возникшие из-за сдвигов (shift)
        df_clean = df.dropna(subset=['Q_lag1', 't_out_delta', 'Q', 't_out']).reset_index(drop=True)
        return df_clean

    def fit(self, df_history):
        """
        Обучение динамической модели под теплофизические свойства конкретного здания
        """
        if len(df_history) < 7:
            raise ValueError("Недостаточно исторических данных для обучения динамической модели (требуется минимум 7 дней).")
            
        df_features = self.prepare_features(df_history)
        
        # Матрица признаков: [Расход_вчера, Т_ул_сегодня, Динамика_Т_ул]
        X = df_features[['Q_lag1', 't_out', 't_out_delta']].values
        y = df_features['Q'].values
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Метрика точности обучения R2
        r2_train = self.model.score(X, y)
        return r2_train

    def predict_next_day(self, last_q, last_t_out, forecast_t_out):
        """
        Математический расчет прогноза на 1 шаг вперед (на завтра)
        last_q: фактический расход сегодня
        last_t_out: температура на улице сегодня
        forecast_t_out: прогноз погоды от синоптиков на завтра
        """
        if not self.is_trained:
            raise ValueError("Модель прогнозирования не обучена.")
            
        t_out_delta_forecast = forecast_t_out - last_t_out
        
        # Вектор признаков для прогноза
        X_pred = np.array([[last_q, forecast_t_out, t_out_delta_forecast]])
        q_forecast = self.model.predict(X_pred)[0]
        
        # Ограничение снизу (физический ноль теплопотребления)
        return max(0.0, float(q_forecast))
