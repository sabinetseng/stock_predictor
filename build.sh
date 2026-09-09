#!/usr/bin/env bash
set -o errexit

# 安裝 Python 依賴
pip install -r requirements.txt

# 執行資料庫遷移
python manage.py migrate

# 收集靜態檔案
python manage.py collectstatic --no-input
