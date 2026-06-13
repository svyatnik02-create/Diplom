# heat_analysis_system/src/ingestion.py
# Модуль парсинга, маппинга столбцов с пробелами и физической валидации

import os
import pandas as pd
import numpy as np

class DataIngestionEngine:
    def __init__(self, config):
        self.cfg = config
        # Точный маппинг под структуру файлов приборов учета МКД
        self.expected_cols = [
            'Дата', 'Время', 'Состояние', 'Отключение', 'Время НС', 
            'T1', 'T2', 'P1', 'P2', 'V1', 'V2', 'M1', 'M2', 'Q', 
            'd T', 'd V', 'd M', 'Небаланс', 'Ти', 'Тост', 'Тхв'
        ]

    def parse_and_validate(self, file_path):
        """Парсинг Excel/CSV-отчета тепловычислителя и физический аудит"""
        file_name = os.path.basename(file_path)
        audit_logs = []
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Ошибка чтения файла {file_name}: {str(e)}")
            
        # Нормализация заголовков от случайных пробелов по краям
        df.columns = [str(col).strip() for col in df.columns]
            
        # Проверка структуры столбцов
        missing_cols = [col for col in self.expected_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Входной файл не содержит обязательных столбцов: {missing_cols}")
            
        total_rows = len(df)
        if total_rows == 0:
            raise ValueError("Файл не содержит данных.")

        # Логический аудит фильтрации по нештатным ситуациям
        status_mask = (df['Время НС'] == 0) & (df['Состояние'].astype(str).str.contains("Допущен|Включен|Раб", case=False, na=False))
        ns_dropped = total_rows - status_mask.sum()
        if ns_dropped > 0:
            audit_logs.append(f"Исключено {ns_dropped} строк из-за наличия НС или некорректного статуса прибора.")
            
        df_valid = df[status_mask].copy()

        # Физическая валидация диапазонов параметров
        limits = self.cfg['validation']['phys_limits']
        
        t_mask = df_valid['T1'].between(limits['t_min'], limits['t_max']) & df_valid['T2'].between(limits['t_min'], limits['t_max'])
        p_mask = df_valid['P1'].between(limits['p_min'], limits['p_max']) & df_valid['P2'].between(limits['p_min'], limits['p_max'])
        q_mask = df_valid['Q'] >= limits['q_min']
        
        invalid_t = len(df_valid) - t_mask.sum()
        invalid_p = len(df_valid) - p_mask.sum()
        invalid_q = len(df_valid) - q_mask.sum()
        
        if invalid_t > 0: audit_logs.append(f"Заменено {invalid_t} аномальных температур (выход за {limits['t_min']}-{limits['t_max']}°C).")
        if invalid_p > 0: audit_logs.append(f"Заменено {invalid_p} аномальных давлений (выход за {limits['p_min']}-{limits['p_max']} бар).")
        if invalid_q > 0: audit_logs.append(f"Заменено {invalid_q} отрицательных значений теплопотребления Q.")

        df_valid.loc[~t_mask, ['T1', 'T2']] = np.nan
        df_valid.loc[~p_mask, ['P1', 'P2']] = np.nan
        df_valid.loc[~q_mask, 'Q'] = np.nan

        # Проверка критического объема потерь данных
        total_nan_q = df_valid['Q'].isna().sum() + (total_rows - len(df_valid))
        missing_rate = total_nan_q / total_rows
        
        if missing_rate > self.cfg['validation']['max_missing_allowed']:
            raise ValueError(f"Файл отклонен: доля пропусков ({missing_rate:.2%}) выше лимита {self.cfg['validation']['max_missing_allowed']:.0%}.")

        # Запись журнала аудита
        audit_path = os.path.join(self.cfg['paths']['audit_dir'], f"audit_{os.path.splitext(file_name)[0]}.txt")
        with open(audit_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(audit_logs))

        return df_valid, audit_logs