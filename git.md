# Git 使用說明檔

本專案已初始化 local git repository。以下說明如何進行版本控管。

---

## ✅ 已完成的設定

- [x] 初始化 git repository (`git init`)
- [x] 設定 `.gitignore`（排除機密檔案、虛擬環境、模型檔）
- [x] 設定使用者資訊
- [x] 完成第一次提交（Initial Commit）
- [x] 建立本說明檔

---

## 🚀 快速開始

### 第一次使用（新電腦）
```bash
# 1. 複製專案
git clone <倉庫網址>

# 2. 進入專案資料夾
cd stock_predictor

# 3. 建立虛擬環境
python -m venv venv
venv\Scripts\activate

# 4. 安裝套件
pip install -r requirements.txt

# 5. 設定機密檔案（從 .env.example 複製）
copy .env.example .env
# 編輯 .env 填入實際的資料庫密碼、API Key

# 6. 執行資料庫遷移
python manage.py migrate

# 7. 啟動開發伺服器
python manage.py runserver
```

---

## 📁 專案結構

```
stock_predictor/
├── .git/                   # Git repository 資料夾（自動產生）
├── .gitignore              # 忽略不追蹤的檔案
├── apps/                   # 各 App 程式碼
│   ├── analysis/           # 相關性分析
│   ├── dashboard/          # 首頁
│   ├── fx_data/            # 美金匯率
│   ├── market_data/        # 大盤指數
│   ├── stocks/             # 股票核心功能
│   └── sync/               # 資料同步
├── stock_predictor/        # Django 專案設定
├── trained_models/         # 訓練好的模型（不追蹤）
├── venv/                   # Python 虛擬環境（不追蹤）
├── .env                    # 機密設定（不追蹤）
├── manage.py               # Django 管理指令
├── requirements.txt        # Python 套件清單
└── README.md               # 專案說明
```

---

## 🔧 Git 基本指令

### 1. 查看目前狀態
```bash
git status
```
顯示哪些檔案已修改、已暫存、或未被追蹤。

### 2. 加入檔案到暫存區（Stage）
```bash
# 加入特定檔案
git add <檔案名稱>

# 加入所有修改的檔案
git add .

# 加入所有檔案（包含新增的）
git add -A
```

### 3. 提交變更（Commit）
```bash
git commit -m "提交訊息"
```
提交訊息應該清楚描述這次做了什麼變更。

### 4. 查看提交歷史
```bash
git log
git log --oneline          # 簡潔模式
git log --oneline --all    # 顯示所有分支
```

### 5. 查看差異
```bash
git diff                   # 工作區 vs 暫存區
git diff --staged          # 暫存區 vs 最後一次提交
git diff HEAD~1            # 最後一次提交 vs 前一次
```

---

## 🌿 分支管理

### 建立與切換分支
```bash
# 建立新分支
git branch <分支名稱>

# 切換到分支
git checkout <分支名稱>

# 建立並切換到新分支
git checkout -b <分支名稱>

# 查看所有分支
git branch -a
```

### 合併分支
```bash
# 先切換到主分支
git checkout main

# 合併其他分支進來
git merge <分支名稱>
```

### 刪除分支
```bash
git branch -d <分支名稱>     # 已合併的分支
git branch -D <分支名稱>     # 強制刪除
```

---

## 🔄 日常工作流程

### 開始工作前
```bash
git pull                   # 遠端倉庫同步到本地（如果有設定遠端）
```

### 開發新功能
```bash
# 1. 建立新功能分支
git checkout -b feature/新功能名稱

# 2. 開發並測試

# 3. 提交變更
git add .
git commit -m "feat: 新增 XXX 功能"

# 4. 切換回主分支
git checkout main

# 5. 合併功能分支
git merge feature/新功能名稱

# 6. 刪除已合併的功能分支
git branch -d feature/新功能名稱
```

### 修復 Bug
```bash
# 1. 建立修復分支
git checkout -b bugfix/問題描述

# 2. 修復並測試

# 3. 提交變更
git add .
git commit -m "fix: 修復 XXX 問題"

# 4. 合併回主分支
git checkout main
git merge bugfix/問題描述
```

---

## 📝 提交訊息規範

建議使用以下格式：

```
<類型>: <簡短描述>

<詳細描述（選擇性）>
```

### 類型說明
| 類型 | 說明 | 範例 |
|------|------|------|
| `feat` | 新功能 | `feat: 新增個股相關性分析圖表` |
| `fix` | 修復 Bug | `fix: 修復預測按鈕無法點擊問題` |
| `docs` | 文件更新 | `docs: 更新 README 安裝說明` |
| `style` | 程式碼格式 | `style: 調整縮排與排版` |
| `refactor` | 重構 | `refactor: 重構特徵計算邏輯` |
| `test` | 測試 | `test: 新增預測功能單元測試` |
| `chore` | 雜務 | `chore: 更新 .gitignore` |

---

## ⚠️ 注意事項

### 不要提交的檔案（已在 .gitignore）
- `.env` - 機密設定（資料庫密碼、API Key）
- `venv/` / `.venv/` - Python 虛擬環境
- `__pycache__/` - Python 快取
- `*.pyc` - Python 編譯檔
- `trained_models/` - 訓練好的模型檔
- `*.sqlite3` - SQLite 資料庫

### 提交前檢查
1. 執行 `git status` 確認要提交的檔案
2. 執行 `git diff` 確認變更內容
3. 確保程式碼可以正常運行
4. 不要提交機密資訊

---

## 🔗 設定遠端倉庫（可選）

如果想要備份到 GitHub/GitLab：

```bash
# 1. 在 GitHub 建立新倉庫

# 2. 設定遠端
git remote add origin https://github.com/<你的帳號>/<倉庫名稱>.git

# 3. 第一次推送
git push -u origin main

# 4. 之後推送
git push
```

---

## 🆘 常見問題

### 1. 想要取消已暫存的檔案
```bash
git restore --staged <檔案名稱>
```

### 2. 想要放棄工作區的修改
```bash
git restore <檔案名稱>
```

### 3. 想要修改最後一次提交的訊息
```bash
git commit --amend -m "新的提交訊息"
```

### 4. 想要暫時儲存修改（切換分支時）
```bash
git stash                  # 暫存
git stash pop              # 恢復
```

### 5. 查看某個檔案的修改歷史
```bash
git log -p <檔案名稱>
```

---

## 📚 快速參考卡

| 動作 | 指令 |
|------|------|
| 初始化 | `git init` |
| 查看狀態 | `git status` |
| 加入檔案 | `git add .` |
| 提交 | `git commit -m "訊息"` |
| 查看歷史 | `git log --oneline` |
| 建立分支 | `git checkout -b <名稱>` |
| 切換分支 | `git checkout <名稱>` |
| 合併 | `git merge <名稱>` |
| 推送 | `git push` |
| 拉取 | `git pull` |

---

## 🎯 建議的專案分支策略

```
main (穩定版本)
  │
  ├── feature/新功能1
  │     └── 開發中...
  │
  ├── feature/新功能2
  │     └── 開發中...
  │
  └── bugfix/修復問題
        └── 修復中...
```

- `main` 分支保持穩定、可執行的版本
- 新功能在 `feature/` 分支開發
- 修復問題在 `bugfix/` 分支處理
- 開發完成後才合併回 `main`

---

*建立日期：2026-09-07*
*作者：專案開發團隊*
