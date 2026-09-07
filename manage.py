#!/usr/bin/env python
"""Django 專案的管理入口，不需修改。"""
import os
import sys


def main():
    # 指定使用的 settings 模組
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stock_predictor.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "無法匯入 Django，請確認已安裝 Django 並啟用虛擬環境。"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
