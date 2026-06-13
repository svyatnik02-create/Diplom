# heat_analysis_system/src/classifier.py
# [Раздел 2.3 Диплома] Экспертная классификация энергоэффективности МКД и формирование рекомендаций

class BuildingClassifier:
    def __init__(self, config):
        self.cfg = config

    def evaluate(self, df, beta_0, beta_1):
        """Оценка интегрального класса эффективности и извлечение причин отклонений"""
        limits = self.cfg['classification']['limits']
        crit_consecutive = self.cfg['classification']['expert_rules']['consecutive_anomalies_crit']
        
        # Расчет нормализованного удельного потребления (Фактическое / Модельное)
        total_fact = df['Q'].sum()
        total_pred = df['q_pred'].sum()
        
        norm_ratio = total_fact / max(total_pred, 1e-4)

        # Первичный скрининг по долевому отклонению
        if norm_ratio <= limits['norm_low']:
            status = "Эффективный"
        elif norm_ratio <= limits['norm_high']:
            status = "Нормативный"
        elif norm_ratio <= limits['warn_high']:
            status = "Предупреждение"
        else:
            status = "Критический"

        # Проверка серии последовательных EWMA аномалий
        consecutive_anomalies = 0
        max_consecutive = 0
        for val in df['is_anomaly']:
            if val:
                consecutive_anomalies += 1
                if consecutive_anomalies > max_consecutive:
                    max_consecutive = consecutive_anomalies
            else:
                consecutive_anomalies = 0

        # Коррекция статуса по цепочкам скрытых сбоев
        if max_consecutive >= crit_consecutive:
            status = "Критический"

        # Генерация экспертных правил (Диагностических директив)
        recommendations = []
        if status in ["Предупреждение", "Критический"]:
            recommendations.append("▶ Требуется внеочередная инструментальная проверка узла смешения (элеватора/АТП).")
            
            # Анализ характера потерь по коэффициентам регрессии
            if abs(beta_1) > 0.15: # Порог чувствительности к погоде
                recommendations.append("⚠ Физическая причина: Высокая чувствительность к наружному воздуху. Рекомендация: теплоизоляция межпанельных швов, фасадов и оконных блоков здания.")
            if beta_0 > (total_pred / len(df)) * 0.5:
                recommendations.append("⚠ Физическая причина: Высокий уровень постоянных потерь. Рекомендация: Проверить циркуляционный контур ГВС на предмет скрытых утечек, свищей и проверить изоляцию подвальных магистралей.")
        else:
            recommendations.append("✔ Режимы теплопотребления здания находятся в рамках статистической нормы. Рекомендуется плановое ТО.")

        if max_consecutive >= crit_consecutive:
            recommendations.append(f"🚨 Зафиксирована затяжная серия из {max_consecutive} суточных EWMA-выбросов, что указывает на раскалибровку автоматики или аварийную разбалансировку стояков.")

        return status, recommendations