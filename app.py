# heat_analysis_system/app.py
# [Раздел 2.1 - 2.3 Диплома] Streamlit UI Дашборд и сквозной аналитический конвейер
# Максимальная сборка: Сравнение МКД, Симулятор нагрузок, Управление СУБД и Настройка ГСОП

import os
import yaml
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime

from src.database import init_db, BuildingMeta, DailyMetrics
from src.ingestion import DataIngestionEngine
from src.preprocessing import DataPreprocessingPipeline
from src.models import RobustHeatModel
from src.anomaly import EWMAAnomalyDetector
from src.classifier import BuildingClassifier

st.set_page_config(layout="wide", page_title="Платформа Анализа Теплопотребления МКД")

# Обеспечение структуры каталогов проекта
os.makedirs("config", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/audit", exist_ok=True)

# Загрузка конфигурации
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Инициализация подключения к СУБД SQLite
Session = init_db(cfg['paths']['database_uri'])

# ====================================================================
# УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ PDF-ОТЧЕТА
# ====================================================================
def generate_pdf_bytes(file_name, status, r2, beta_0, beta_1, recommendations, tech_lines):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.add_font("Arial", "", "C:/Windows/Fonts/arial.ttf", uni=True)
    pdf.set_font("Arial", size=14)
    
    pdf.cell(190, 10, txt="Инженерный отчет комплексного энергоаудита МКД", ln=1, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", size=11)
    pdf.cell(190, 8, txt=f"Файл объекта автоматизации: {file_name}", ln=1)
    pdf.cell(190, 8, txt=f"Интегральный класс эффективности: {status}", ln=1)
    pdf.cell(190, 8, txt=f"Метрика адекватности модели (Робастный R2): {r2:.4f}", ln=1)
    pdf.cell(190, 8, txt=f"Постоянные фоновые потери теплоэнергии здания: {beta_0:.4f} Гкал/сут", ln=1)
    pdf.cell(190, 8, txt=f"Динамический погодозависимый коэффициент: {beta_1:.4f}", ln=1)
    pdf.cell(190, 5, txt="-----------------------------------------------------------------------------------------", ln=1)
    pdf.ln(2)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(190, 8, txt="ТЕХНИЧЕСКИЙ ЖУРНАЛ ФИКСАЦИИ ПОВРЕЖДЕНИЙ И АНОМАЛИЙ:", ln=1)
    pdf.ln(2)
    
    pdf.set_font("Arial", size=9)
    for line in tech_lines:
        pdf.multi_cell(190, 6, txt=line)
        pdf.ln(1)
    
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.cell(190, 8, txt="СТРАТЕГИЧЕСКИЕ ДИРЕКТИВЫ НАСТРОЙКИ АВТОМАТИКИ ИТП:", ln=1)
    pdf.ln(2)
    
    for rec in recommendations:
        clean_rec = rec.replace("🚨", "[КРИТИЧЕСКИ]").replace("⚠", "[ВНИМАНИЕ]").replace("ℹ", "[ИНФО]")
        pdf.multi_cell(190, 7, txt=clean_rec)
        pdf.ln(2)
        
    raw_output = pdf.output(dest='S')
    return raw_output.encode('latin1') if isinstance(raw_output, str) else bytes(raw_output)


# ====================================================================
# БОКОВАЯ ПАНЕЛЬ И НАСТРОЙКА РЕГИОНАЛЬНЫХ ПАРАМЕТРОВ (СП 50.13330)
# ====================================================================
st.sidebar.title("🧭 Панель управления АС")
app_mode = st.sidebar.radio(
    "Выберите модуль системы:",
    [
        "🔄 Проведение нового анализа", 
        "🗄️ Архив технических аудитов", 
        "📊 Сравнение объектов МКД",
        "🔮 Симулятор тепловых нагрузок ИТП"
    ]
)

st.sidebar.write("---")
st.sidebar.subheader("⚙️ Параметры ГСОП региона")
t_base_user = st.sidebar.slider("Целевая T_вн в квартирах (°C)", 18.0, 24.0, float(cfg['region_settings']['t_base_room']), 0.5)
t_op_user = st.sidebar.slider("Порог включения отопления T_от (°C)", 0.0, 12.0, float(cfg['region_settings']['t_op']), 0.5)

cfg['region_settings']['t_base_room'] = t_base_user
cfg['region_settings']['t_op'] = t_op_user


# ====================================================================
# МОДУЛЬ 1: ПРОВЕДЕНИЕ НОВОГО АНАЛИЗА
# ====================================================================
if app_mode == "🔄 Проведение нового анализа":
    st.title("🏛️ Экспресс-анализ теплопотребления МКД")
    st.subheader("Сквозной аналитический конвейер статистического контроля")

    uploaded_file = st.file_uploader("Загрузить отчет тепловычислителя (Excel/CSV)", type=["xlsx", "xls", "csv"])
    weather_file = st.file_uploader("📋 [Опционально] Загрузить CSV с погодой (разделитель ';')", type=["csv"])

    if uploaded_file is not None:
        raw_path = os.path.join(cfg['paths']['raw_dir'], uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.info(f"Файл {uploaded_file.name} сохранён. Расчет ГСОП ведется по внутренней температуре {t_base_user}°C.")

        weather_df = None
        if weather_file is not None:
            weather_df = pd.read_csv(weather_file, sep=';')

        ingester = DataIngestionEngine(cfg)
        preprocessor = DataPreprocessingPipeline(cfg)
        model_core = RobustHeatModel(cfg)
        detector = EWMAAnomalyDetector(cfg)
        classifier = BuildingClassifier(cfg)

        try:
            df_valid, logs = ingester.parse_and_validate(raw_path)
            df_daily = preprocessor.process(df_valid, weather_df=weather_df)
            df_modeled = model_core.fit(df_daily)
            df_analyzed = detector.analyze(df_modeled)
            status, recommendations = classifier.evaluate(df_analyzed, model_core.beta_0, model_core.beta_1)

            st.success("✅ Аналитический конвейер успешно выполнен.")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Статус Энергоэффективности", status)
            col2.metric("Базовые потери тепла (β₀)", f"{model_core.beta_0:.3f} Гкал/сут")
            col3.metric("Чувствительность к морозу (β₁)", f"{model_core.beta_1:.4f}")
            col4.metric("Адекватность модели (Робастный R²)", f"{model_core.r2:.2%}")

            st.write("### 📊 Интерактивный графический анализ")
            fig_huber = go.Figure()
            fig_huber.add_trace(go.Scatter(x=df_analyzed['t_out'], y=df_analyzed['Q'], mode='markers', name='Фактические точки'))
            df_sorted = df_analyzed.sort_values('t_out')
            fig_huber.add_trace(go.Scatter(x=df_sorted['t_out'], y=df_sorted['q_pred'], mode='lines', name='Модель Хубера', line=dict(color='crimson', width=3)))
            st.plotly_chart(fig_huber, use_container_width=True)

            fig_ewma = go.Figure()
            fig_ewma.add_trace(go.Scatter(x=df_analyzed['Date'], y=df_analyzed['ewma_val'], mode='lines+markers', name='Статистика EWMA'))
            fig_ewma.add_trace(go.Scatter(x=df_analyzed['Date'], y=df_analyzed['ewma_ucl'], mode='lines', name='UCL', line=dict(dash='dash')))
            fig_ewma.add_trace(go.Scatter(x=df_analyzed['Date'], y=df_analyzed['ewma_lcl'], mode='lines', name='LCL', line=dict(dash='dash')))
            anomalies_points = df_analyzed[df_analyzed['is_anomaly']]
            fig_ewma.add_trace(go.Scatter(x=anomalies_points['Date'], y=anomalies_points['ewma_val'], mode='markers', name='🚨 Выброс', marker=dict(color='red', size=10, symbol='x')))
            st.plotly_chart(fig_ewma, use_container_width=True)

            st.write("---")
            st.write("### 🧠 Технический рапорт повреждений и аномалий")
            
            anomalous_days = df_analyzed[df_analyzed['is_anomaly']].copy()
            technical_summary_pdf = []

            if len(anomalous_days) == 0:
                st.success("🎉 Скрытых гидравлических и температурных аномалий не обнаружено.")
                technical_summary_pdf.append("Скрытых технологических аномалий за период не обнаружено.")
            else:
                for idx, row in anomalous_days.iterrows():
                    date_str = row['Date'].strftime('%d.%m.%Y')
                    q_fact, q_pred, t_out, ewma_val = float(row['Q']), float(row['q_pred']), float(row['t_out']), float(row['ewma_val'])
                    ucl, lcl = float(row['ewma_ucl']), float(row['ewma_lcl'])
                    
                    if ewma_val > ucl:
                        severity = "КРИТИЧЕСКАЯ" if (ewma_val - ucl) > 1.5 else "ЗНАЧИТЕЛЬНАЯ"
                        anomaly_type = f"Превышение расхода (Перетоп). Перерасход: {q_fact - q_pred:.3f} Гкал/сут."
                        border_color = "#dc3545"
                    else:
                        severity = "ВЫСОКАЯ" if (lcl - ewma_val) > 1.5 else "УМЕРЕННАЯ"
                        anomaly_type = f"Падение теплопотребления. Дефицит: {q_pred - q_fact:.3f} Гкал/сут."
                        border_color = "#28a745"
                    
                    hardware_context = ""
                    try:
                        raw_row = df_valid[pd.to_datetime(df_valid['Дата']) == row['Date']].iloc[0]
                        hardware_context = f"| Параметры: Т1={raw_row['T1']}°C, Т2={raw_row['T2']}°C, Р1={raw_row['P1']} бар, Р2={raw_row['P2']} bar"
                    except:
                        pass
                    
                    report_line = f"Дата: {date_str} | T_out: {t_out}°C | {anomaly_type} | Степень: {severity}"
                    technical_summary_pdf.append(report_line)
                    
                    st.markdown(f"""
                    <div style="padding:10px; margin-bottom:8px; border-radius:5px; border-left:5px solid {border_color}; background-color:rgba(0,0,0,0.02); font-size:13px;">
                        <strong>📅 Дата:</strong> {date_str} | Уровень угрозы: <span style="color:{border_color};font-weight:bold;">{severity}</span><br/>
                        <strong>🔍 Инцидент:</strong> {anomaly_type} <span style="font-family:monospace; color:#666;">{hardware_context}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Сохранение результатов в БД SQLite СУБД через ORM
            session = Session()
            try:
                existing_meta = session.query(BuildingMeta).filter_by(file_name=uploaded_file.name).first()
                if existing_meta:
                    session.query(DailyMetrics).filter_by(building_id=existing_meta.id).delete()
                    session.delete(existing_meta)
                    session.commit()
                
                meta_record = BuildingMeta(
                    file_name=uploaded_file.name, beta_0=model_core.beta_0,
                    beta_0_ci_low=model_core.beta_0_ci[0], beta_0_ci_high=model_core.beta_0_ci[1],
                    beta_1=model_core.beta_1, beta_1_ci_low=model_core.beta_1_ci[0], beta_1_ci_high=model_core.beta_1_ci[1],
                    r2_score=model_core.r2, huber_delta=model_core.delta, status=status
                )
                session.add(meta_record)
                session.commit()

                daily_records = []
                for _, row in df_analyzed.iterrows():
                    metric_rec = DailyMetrics(
                        building_id=meta_record.id, date=str(row['Date'].date()), t_out=float(row['t_out']),
                        q_fact=float(row['Q']), q_pred=float(row['q_pred']), residual=float(row['residual']),
                        ewma_val=float(row['ewma_val']), ewma_ucl=float(row['ewma_ucl']), ewma_lcl=float(row['ewma_lcl']),
                        is_anomaly=bool(row['is_anomaly']), is_interpolated=bool(row['is_interpolated'])
                    )
                    daily_records.append(metric_rec)
                session.bulk_save_objects(daily_records)
                session.commit()
            except Exception as db_err:
                session.rollback()
                raise db_err
            finally:
                session.close()

            st.write("### 📄 Экспорт аналитического заключения")
            pdf_bytes = generate_pdf_bytes(uploaded_file.name, status, model_core.r2, model_core.beta_0, model_core.beta_1, recommendations, technical_summary_pdf)
            st.download_button(
                label="📥 Скачать развернутый инженерный PDF-отчет",
                data=pdf_bytes,
                file_name=f"report_{os.path.splitext(uploaded_file.name)[0]}.pdf",
                mime="application/pdf"
            )

        except Exception as ex:
            st.error(f"Критический сбой конвейера: {str(ex)}")


# ====================================================================
# МОДУЛЬ 2: АРХИВ ТЕХНИЧЕСКИХ АУДИТОВ
# ====================================================================
elif app_mode == "🗄️ Архив технических аудитов":
    st.title("🗄️ Электронный журнал технических аудитов МКД")
    
    session = Session()
    try:
        history_records = session.query(BuildingMeta).order_by(BuildingMeta.id.desc()).all()
        
        if not history_records:
            st.info("В базе данных SQLite пока нет сохраненных расчетов.")
        else:
            building_map = {f"ID {rec.id} | {rec.file_name} ({rec.status.strip()})": rec for rec in history_records}
            selected_key = st.selectbox("🎯 Выберите объект автоматизации для просмотра:", list(building_map.keys()))
            selected_meta = building_map[selected_key]
            
            if st.button("🗑️ Удалить эту запись из базы данных"):
                session.query(DailyMetrics).filter_by(building_id=selected_meta.id).delete()
                session.delete(selected_meta)
                session.commit()
                st.success("Запись успешно удалена из архива. Обновите страницу.")
                st.rerun()

            st.write("---")
            metrics_records = session.query(DailyMetrics).filter_by(building_id=selected_meta.id).order_by(DailyMetrics.date).all()
            
            data_dict = {
                "Date": [datetime.strptime(m.date, "%Y-%m-%d") for m in metrics_records],
                "t_out": [m.t_out for m in metrics_records],
                "Q": [m.q_fact for m in metrics_records],
                "q_pred": [m.q_pred for m in metrics_records],
                "ewma_val": [m.ewma_val for m in metrics_records],
                "ewma_ucl": [m.ewma_ucl for m in metrics_records],
                "ewma_lcl": [m.ewma_lcl for m in metrics_records],
                "is_anomaly": [m.is_anomaly for m in metrics_records]
            }
            df_hist = pd.DataFrame(data_dict)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Архивный класс", selected_meta.status)
            col2.metric("Потери β₀", f"{selected_meta.beta_0:.3f} Гкал/сут")
            col3.metric("Погодозависимость β₁", f"{selected_meta.beta_1:.4f}")
            col4.metric("Точность R²", f"{selected_meta.r2_score:.2%}")
            
            hist_tech_summary_pdf = []
            st.write("##### 📋 Выявленные за период отклонения:")
            df_anom_only = df_hist[df_hist['is_anomaly']].copy()
            if len(df_anom_only) == 0:
                st.success("Технологических аномалий на объекте не зафиксировано.")
                hist_tech_summary_pdf.append("Скрытых аномалий за период не обнаружено.")
            else:
                for idx, row in df_anom_only.iterrows():
                    d_str = row['Date'].strftime('%d.%m.%Y')
                    q_f, q_p, t_o, ewma_v = float(row['Q']), float(row['q_pred']), float(row['t_out']), float(row['ewma_val'])
                    u_l, l_l = float(row['ewma_ucl']), float(row['ewma_lcl'])
                    sev = "КРИТИЧЕСКАЯ" if (ewma_v - u_l) > 1.5 or (l_l - ewma_v) > 1.5 else "ШТАТНАЯ"
                    
                    a_type = "Перерасход тепловой энергии (Перетоп)" if ewma_v > u_l else "Зажатие расхода (Недогрев)"
                    r_line = f"Дата: {d_str} | T_out: {t_o}°C | {a_type} | Критичность: {sev}"
                    hist_tech_summary_pdf.append(r_line)
                    st.text(r_line)

            hist_recs = ["Архивное извлечение. Параметры отопительного периода стабильны."]
            pdf_bytes = generate_pdf_bytes(selected_meta.file_name, selected_meta.status, selected_meta.r2_score, selected_meta.beta_0, selected_meta.beta_1, hist_recs, hist_tech_summary_pdf)
            st.download_button(
                label="📥 Восстановить PDF-отчет из архива",
                data=pdf_bytes,
                file_name=f"recovered_report_{selected_meta.id}.pdf",
                mime="application/pdf"
            )
            
    except Exception as db_err:
        st.error(f"Ошибка БД: {str(db_err)}")
    finally:
        session.close()


# ====================================================================
# МОДУЛЬ 3: СРАВНЕНИЕ ОБЪЕКТОВ МКД МЕЖДУ СОБОЙ (ИСПРАВЛЕНО!)
# ====================================================================
elif app_mode == "📊 Сравнение объектов МКД":
    st.title("📊 Модуль сравнительного анализа МКД")
    st.subheader("Сравнение энергетических профилей зданий для выявления критических потерь")
    
    session = Session()
    try:
        all_buildings = session.query(BuildingMeta).all()
        if len(all_buildings) < 2:
            st.info("⚠️ Для сравнения необходимо, чтобы в базе данных было сохранено как минимум 2 объекта. Загрузите файлы разных домов.")
        else:
            b_options = {f"ID {b.id} | {b.file_name}": b for b in all_buildings}
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                choice1 = st.selectbox("🏠 Выберите Здание №1:", list(b_options.keys()), index=0)
                meta1 = b_options[choice1]
            with col_b2:
                choice2 = st.selectbox("🏠 Выберите Здание №2:", list(b_options.keys()), index=1 if len(all_buildings) > 1 else 0)
                meta2 = b_options[choice2]
                
            st.write("---")
            st.markdown("#### ⚖️ Сравнительная таблица коэффициентов робастных моделей")
            
            compare_df = pd.DataFrame({
                "Параметр энергоэффективности": [
                    "Интегральный статус", 
                    "Фоновые тепловые потери (β₀), Гкал/сут", 
                    "Погодозависимый коэффициент (β₁)", 
                    "Качество аппроксимации данных (R²)"
                ],
                f"Объект 1 ({meta1.file_name[:20]})": [meta1.status, f"{meta1.beta_0:.3f}", f"{meta1.beta_1:.4f}", f"{meta1.r2_score:.2%}"],
                f"Объект 2 ({meta2.file_name[:20]})": [meta2.status, f"{meta2.beta_0:.3f}", f"{meta2.beta_1:.4f}", f"{meta2.r2_score:.2%}"]
            })
            st.table(compare_df)
            
            # Визуализация сравнения потерь
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='Фоновые потери β₀', x=[meta1.file_name[:15], meta2.file_name[:15]], y=[meta1.beta_0, meta2.beta_0], marker_color='indianred'))
            fig_comp.add_trace(go.Bar(name='Погодозависимость β₁', x=[meta1.file_name[:15], meta2.file_name[:15]], y=[meta1.beta_1, meta2.beta_1], marker_color='lightseagreen'))
            
            # ИСПРАВЛЕНО: Заменено ошибочное bgroupmode на корректное barmode
            fig_comp.update_layout(title="Сравнительный анализ структурных коэффициентов теплопотребления", barmode='group', template='plotly_white')
            st.plotly_chart(fig_comp, use_container_width=True)
            
    except Exception as ex:
        st.error(f"Ошибка модуля сравнения: {str(ex)}")
    finally:
        session.close()


# ====================================================================
# МОДУЛЬ 4: СИМУЛЯТОР ТЕПЛОВЫХ НАГРУЗОК ИТП
# ====================================================================
elif app_mode == "🔮 Симулятор тепловых нагрузок ИТП":
    st.title("🔮 Модуль прогнозирования и предиктивного моделирования нагрузки")
    st.subheader("Расчет ожидаемого часового и суточного расхода тепла по уравнению Хубера")
    
    session = Session()
    try:
        all_buildings = session.query(BuildingMeta).all()
        if not all_buildings:
            st.info("База архивных моделей пуста. Сначала выполните расчет в Модуле №1.")
        else:
            b_options = {f"{b.file_name}": b for b in all_buildings}
            selected_b = st.selectbox("🏠 Выберите МКД для имитационного моделирования:", list(b_options.keys()))
            b_meta = b_options[selected_b]
            
            st.write("---")
            st.markdown("#### 🎛️ Панель имитационного моделирования экстремальных условий")
            
            t_sim = st.slider("Задайте планируемую температуру наружного воздуха T_out (°C):", -45.0, 15.0, -10.0, 0.5)
            
            t_diff = t_base_user - t_sim
            if t_sim <= t_op_user:
                q_simulated = b_meta.beta_0 + b_meta.beta_1 * t_diff
            else:
                q_simulated = b_meta.beta_0
                
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric(f"Прогноз расхода при {t_sim}°C", f"{q_simulated:.3f} Гкал/сут")
            col_s2.metric("Ожидаемая часовая нагрузка", f"{q_simulated/24:.4f} Гкал/ч")
            col_s3.metric("Статус здания в базе", b_meta.status.strip())
            
            st.info(f"ℹ️ Данный расчет позволяет диспетчеру определить, сколько тепловой энергии запросит здание при наступлении сильных морозов ({t_sim}°C), для исключения аварийного гидравлического голодания ИТП.")
            
    except Exception as ex:
        st.error(f"Ошибка модуля симуляции: {str(ex)}")
    finally:
        session.close()