# heat_analysis_system/src/anomaly.py
# [Раздел 2.3 Диплома] Детекция тепловых аномалий методом EWMA контрольных карт на остатках модели

import pandas as pd
import numpy as np

class EWMAAnomalyDetector:
    def __init__(self, config):
        self.cfg = config

    def analyze(self, df):
        """Расчет скользящих пределов EWMA карты и маркировка статистических выбросов"""
        span = self.cfg['anomaly']['ewma']['span']
        k_sigma = self.cfg['anomaly']['ewma']['k_sigma']
        
        # Перевод параметра span в коэффициент затухания альфа
        alpha = 2.0 / (span + 1)
        
        # Расчет экспоненциально взвешенного скользящего среднего на остатках (невязках) регрессии
        df['ewma_val'] = df['residual'].ewm(alpha=alpha, adjust=False).mean()
        
        # Историческая дисперсия остатков для расчета динамических труб контроля
        std_res = df['residual'].std()
        if np.isnan(std_res) or std_res == 0:
            std_res = 1.0

        # Моделирование изменяющихся во времени пределов EWMA
        # Пределы сужаются на первых наблюдениях и стабилизируются к k * sigma
        t_steps = np.arange(1, len(df) + 1)
        ewma_std = std_res * np.sqrt((alpha / (2 - alpha)) * (1 - (1 - alpha) ** (2 * t_steps)))
        
        df['ewma_ucl'] = k_sigma * ewma_std
        df['ewma_lcl'] = -k_sigma * ewma_std
        
        # Факт аномалии - выход статистики за верхний или нижний предел
        df['is_anomaly'] = (df['ewma_val'] > df['ewma_ucl']) | (df['ewma_val'] < df['ewma_lcl'])
        
        return df
