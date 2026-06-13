# heat_analysis_system/src/preprocessing.py
# Календарный анализ, импутация и расчет ГСОП для готовых суточных рядов
# Полная совместимость с pandas 2.1+ (использование .ffill() и .bfill())

import pandas as pd
import numpy as np

class DataPreprocessingPipeline:
    def __init__(self, config):
        self.cfg = config

    def process(self, df_valid, weather_df=None):
        """Обработка суточного временного ряда"""
        df_valid['Parsed_Date'] = pd.to_datetime(df_valid['Дата'])
        
        daily = df_valid.set_index('Parsed_Date')
        daily = daily[['T1', 'T2', 'P1', 'P2', 'Q']].tz_localize(None)
        
        # Гарантия непрерывности календаря
        full_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='D')
        daily = daily.reindex(full_idx)
        daily.index.name = 'Date'

        daily['is_interpolated'] = daily['Q'].isna()

        # Двухуровневая импутация пропусков
        missing_blocks = daily['Q'].isna().astype(int).groupby(daily['Q'].notna().cumsum()).cumsum()
        
        # 1. Линейная интерполяция (малые окна <= 2 суток)
        short_gap_mask = daily['Q'].isna() & (missing_blocks <= 2)
        if short_gap_mask.any():
            daily['Q'] = daily['Q'].interpolate(method='linear', limit=2)
            
        # 2. Модельная профильная импутация (окна > 2 суток)
        if daily['Q'].isna().any():
            daily['day_of_week'] = daily.index.dayofweek
            median_profile = daily.groupby('day_of_week')['Q'].transform('median')
            daily['Q'] = daily['Q'].fillna(median_profile)
            daily['Q'] = daily['Q'].bfill().ffill()

        # Интеграция климатических параметров
        daily = self._integrate_weather(daily, weather_df)

        # Расчет вклада в ГСОП по СП 50.13330.2012
        t_base = self.cfg['region_settings']['t_base_room']
        t_op = self.cfg['region_settings']['t_op']
        daily['GSOP_contrib'] = daily['t_out'].apply(lambda x: (t_base - x) if x <= t_op else 0.0)

        # Формирование лаговых переменных
        daily['Q_lag1'] = daily['Q'].shift(1).bfill()
        daily['Q_lag3'] = daily['Q'].shift(3).bfill()
        daily['is_weekend'] = daily.index.dayofweek.isin([5, 6]).astype(int)

        return daily.reset_index()

    def _integrate_weather(self, daily_df, weather_df):
        """Интеграция внешнего фактора погоды или генерация синтетического климата"""
        if weather_df is not None:
            weather_df.columns = [str(col).strip() for col in weather_df.columns]
            
            if 'Date' not in weather_df.columns or 't_out' not in weather_df.columns:
                raise KeyError(f"Файл погоды должен содержать столбцы 'Date' и 't_out'. Найдено: {list(weather_df.columns)}")
            
            # Парсинг дат формата ДД.ММ.ГГГГ
            weather_df['Date'] = pd.to_datetime(weather_df['Date'], format='%d.%m.%Y', errors='coerce')
            weather_df = weather_df.dropna(subset=['Date']).set_index('Date')
            
            # Очистка температур от запятых
            if weather_df['t_out'].dtype == object:
                weather_df['t_out'] = weather_df['t_out'].astype(str).str.replace(',', '.').astype(float)
            
            daily_df = daily_df.join(weather_df[['t_out']], how='left')
            daily_df['t_out'] = daily_df['t_out'].ffill().bfill()
        else:
            # Генерация климата при отсутствии файла под ГСОП Хакасии
            np.random.seed(42)
            n_days = len(daily_df)
            time_steps = np.linspace(0, np.pi, n_days)
            synthetic_t = -14.0 - 19.0 * np.sin(time_steps) + np.random.normal(0, 3.0, n_days)
            
            t_base = self.cfg['region_settings']['t_base_room']
            current_gsop = np.sum([t_base - t for t in synthetic_t if t <= self.cfg['region_settings']['t_op']])
            ratio = self.cfg['region_settings']['fallback_weather']['gsop_target'] / max(current_gsop, 1)
            
            daily_df['t_out'] = t_base - (t_base - synthetic_t) * ratio
            
        return daily_df