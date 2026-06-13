# heat_analysis_system/src/database.py
# [Раздел 2.1 Диплома] Схема хранения метаданных и результатов анализа средствами SQLAlchemy

import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class BuildingMeta(Base):
    """Метаданные МКД и интегральные результаты калибровки модели"""
    __tablename__ = 'building_meta'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String, unique=True, nullable=False)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)
    beta_0 = Column(Float, nullable=True)          # Условно-постоянные потери
    beta_0_ci_low = Column(Float, nullable=True)
    beta_0_ci_high = Column(Float, nullable=True)
    beta_1 = Column(Float, nullable=True)          # Погодозависимый коэффициент
    beta_1_ci_low = Column(Float, nullable=True)
    beta_1_ci_high = Column(Float, nullable=True)
    r2_score = Column(Float, nullable=True)        # Коэффициент детерминации
    huber_delta = Column(Float, nullable=True)     # Откалиброванный параметр дельта
    status = Column(String, nullable=False)         # Итоговый статус энергоэффективности

class DailyMetrics(Base):
    """Суточные очищенные и расчетные метрики теплопотребления"""
    __tablename__ = 'daily_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    building_id = Column(Integer, ForeignKey('building_meta.id'), nullable=False)
    date = Column(String, nullable=False)          # Дата в формате YYYY-MM-DD
    t_out = Column(Float, nullable=False)          # Температура наружного воздуха
    q_fact = Column(Float, nullable=False)         # Фактическое теплопотребление
    q_pred = Column(Float, nullable=True)          # Модельное (базовое) потребление
    residual = Column(Float, nullable=True)        # Остаток (невязка)
    ewma_val = Column(Float, nullable=True)        # Статистика EWMA
    ewma_ucl = Column(Float, nullable=True)        # Верхний контрольный предел
    ewma_lcl = Column(Float, nullable=True)        # Нижний контрольный предел
    is_anomaly = Column(Boolean, default=False)
    is_interpolated = Column(Boolean, default=False)

def init_db(database_uri):
    engine = create_engine(database_uri, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)