@echo off
:: Устанавливаем кодировку UTF-8 для корректного отображения кириллицы в консоли Windows
chcp 65001 > nul

title Heat Analysis System LAN Runner
echo ====================================================================
echo Запуск локального окружения системы анализа теплопотребления МКД...
echo ====================================================================

if not exist venv (
    echo [INFO] Виртуальное окружение venv не найдено. Создание venv...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo [INFO] Обновление менеджера пакетов pip и установка зависимостей...
echo [WARN] Используются параметры обхода корпоративного SSL-контроля...

:: Запуск установки с явным указанием доверенных хостов (минуя проверку SSL)
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

echo [INFO] Проверка конфигурации...
if not exist config\config.yaml (
    echo [ERROR] Файл config\config.yaml отсутствует. Проверьте структуру папок!
    pause
    exit
)

echo [INFO] Запуск веб-сервера Streamlit на сетевом интерфейсе 0.0.0.0 (LAN)...
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

pause