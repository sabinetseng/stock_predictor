"""
Django 專案設定檔骨架。
機密資料（DB 密碼等）一律從 .env 讀取，不寫死在這裡。
"""
from pathlib import Path
import os
import shutil
import tempfile
import environ
import dj_database_url

# ---------------------------------------------------------------------------
# SSL 憑證路徑修復：專案放在含中文/非 ASCII 的資料夾時（例：20260807_股票專案），
# yfinance 底層的 curl 無法讀取中文 CA 憑證路徑，會拋 curl(77) 並讓「所有」
# Yahoo Finance 下載回傳空值（指數/匯率同步被誤判為「沒有新資料」）。
# 解法：把 certifi 的 cacert.pem 複製到純 ASCII 的暫存路徑，並以環境變數指定。
# 必須在任何會用到網路的模組匯入前執行（settings 是所有入口的最上游）。
# ---------------------------------------------------------------------------
try:
    import certifi

    _ca_src = certifi.where()
    _ca_dst = os.path.join(tempfile.gettempdir(), "stockproj_cacert.pem")
    if not os.path.exists(_ca_dst) or os.path.getmtime(_ca_dst) < os.path.getmtime(_ca_src):
        shutil.copyfile(_ca_src, _ca_dst)
    os.environ["SSL_CERT_FILE"] = _ca_dst          # libcurl / curl_cffi 讀這個
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca_dst)
except Exception:  # noqa: BLE001 - 憑證修復失敗不該擋住服務啟動
    pass

# 專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent

# 讀取 .env 檔案（密碼、DEBUG 開關等機密設定）
env = environ.Env(
    DEBUG=(bool, False),
)
environ.Env.read_env(BASE_DIR / ".env")

SECRET_KEY = env("DJANGO_SECRET_KEY", default="請在正式環境改成隨機字串")
DEBUG = env("DEBUG")
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])
ALLOWED_HOSTS = ['192.168.1.102', '127.0.0.1', 'localhost']

# 已安裝的 App
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # 專案自訂 App
    "apps.dashboard",
    "apps.stocks",
    "apps.fx_data",
    "apps.market_data",
    "apps.analysis",
    "apps.sync",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "stock_predictor.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # 各 App 內的 templates/ 資料夾會被自動掃描（APP_DIRS=True）
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "stock_predictor.wsgi.application"

# 資料庫設定：使用 PostgreSQL（Render 部署時使用 DATABASE_URL 環境變數）
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'stock_predictor',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Render 部署時自動使用 DATABASE_URL 環境變數
db_from_env = dj_database_url.config(conn_max_age=600)
if db_from_env:
    DATABASES['default'].update(db_from_env)

# MariaDB資料庫設定
#  pip install mysqlclient

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.mysql',  # MariaDB 共用 MySQL 的後端
#         'NAME': 'stock_predictor',           # 在 MariaDB 中建立好的資料庫名稱
#         'USER': 'root',                      # 你的 MariaDB 使用者名稱 (例如 root)
#         'PASSWORD': 'your_password',         # 你的 MariaDB 密碼
#         'HOST': 'localhost',                 # 資料庫主機位置
#         'PORT': '3306',                      # MariaDB 預設連接埠
#     }
# }


AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# 語系與時區：台股系統使用繁體中文、台北時區
LANGUAGE_CODE = "zh-hant"
TIME_ZONE = "Asia/Taipei"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------
# 專案自訂設定（爬蟲、模型相關參數）
# ---------------------------------------------------------

# 訓練 / 驗證資料的時間範圍，供 stocks / analysis app 共用
# 訓練 / 驗證資料的時間範圍「預設值」，供表單初始值與 fallback 使用。
# 實際每次訓練的區間以 apps.stocks.models.ModelTrainingRun 記錄的為準，
# 前端「訓練/驗證區間設定」區塊可手動調整，不強制等於這裡的預設值。
TRAIN_DATE_START = "2018-01-01"
TRAIN_DATE_END = "2024-12-31"
VALIDATION_DATE_START = "2025-01-01"
VALIDATION_DATE_END = "2025-12-31"

# 標籤定義使用的門檻值（3%）
LABEL_RETURN_THRESHOLD = 0.03
LABEL_LOOKAHEAD_TRADING_DAYS = 20

# 標籤定義預設採用進階雙關卡版本（超額報酬 + 停損），預設啟用
USE_ADVANCED_LABEL_GATE = True

# 交易日行事曆資料來源：優先使用 FinMind，若 FinMind 無提供則退回其他公開資料源
TRADING_CALENDAR_SOURCE = "finmind"

# 主要預測模型：LightGBM（處理不平衡標籤、訓練速度、feature importance 皆較適合本專案）
# XGBoost 列為次要對照模型，非必要不啟用
PRIMARY_MODEL = "lightgbm"
SECONDARY_MODEL = "xgboost"  # 選用，供效果比較

# 停損關卡（僅在 USE_ADVANCED_LABEL_GATE = True 時生效）
STOP_LOSS_RATIO = 0.95

# 大盤對照指數代碼（用於相對強弱勢、Beta 計算）
BENCHMARK_INDEX_CODE = "^TWII"
BENCHMARK_STOCK_CODE = "2330"

# 跨市場 / 總經資料代碼
CROSS_MARKET_TICKERS = {
    "sox": "^SOX",
    "vix": "^VIX",
    "usd_twd": "TWD=X",
}

# 注意：FinMind（免費版）與 yfinance 皆不需要 API Token，不需在此設定金鑰。
# 若未來改用 FinMind 付費版，Token 請放入 .env，並用 env("FINMIND_TOKEN") 讀取。

# ---------------------------------------------------------
# Celery（非同步訓練用，預設關閉）
# ---------------------------------------------------------
# TRAINING_ASYNC=False 時，「建立訓練紀錄」API 會像現在一樣同步等訓練跑完才回應。
# 設成 True 之後，API 會立刻回應「已排入訓練佇列」，實際訓練交給 Celery worker 背景執行，
# 但這需要額外啟動 Redis 與 celery worker（見 stock_predictor/celery.py 的說明），
# 沒啟動的話請維持 False，否則訓練任務會卡住不會執行。
TRAINING_ASYNC = env.bool("TRAINING_ASYNC", default=False)

CELERY_BROKER_URL = env("CELERY_BROKER_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default="redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
