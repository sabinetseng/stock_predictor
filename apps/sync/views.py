from django.http import JsonResponse
from django.utils import timezone
from .models import DataSyncLog


def _tw_iso(dt):
    """UTC → 台北時間（settings.TIME_ZONE = Asia/Taipei）的 ISO 字串，給前端直接顯示。"""
    if not dt:
        return None
    return timezone.localtime(dt).isoformat() if timezone.is_aware(dt) else dt.isoformat()


def check_stock_data(request, stock_code):
    """
    「檢查是否有新資料」按鈕。
    TODO: 呼叫 apps.stocks.services.check_and_sync_stock_data(stock_code)
    """
    return JsonResponse({
        "success": True,
        "message": "尚未實作，這是骨架回應",
        "data": {"stock_code": stock_code},
    })


def check_fx_data(request):
    """
    「檢查美金是否有新資料」按鈕。
    TODO: 呼叫 apps.fx_data.services.check_and_sync_fx_data()
    """
    return JsonResponse({
        "success": True,
        "message": "尚未實作，這是骨架回應",
        "data": {},
    })


def recent_logs(request):
    """回傳最近的同步紀錄，供前端顯示（可選功能）。"""
    logs = DataSyncLog.objects.all()[:20]
    data = [
        {
            "source_type": log.source_type,
            "target_code": log.target_code,
            "status": log.status,
            "message": log.message,
            "triggered_at": _tw_iso(log.triggered_at),
        }
        for log in logs
    ]
    return JsonResponse({"success": True, "message": "", "data": data})
