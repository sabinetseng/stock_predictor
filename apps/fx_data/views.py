from django.http import JsonResponse
from . import services
from .models import UsdTwdRate


def check_new_fx_data(request):
    """
    「檢查美金是否有新資料」按鈕呼叫的 API。
    邏輯：比對資料庫最後日期 -> 有新資料才 upsert -> 寫 sync log。
    """
    result = services.check_and_sync_fx_data()
    status_code = 200 if result["success"] else 500
    return JsonResponse(result, status=status_code)


def fx_history(request):
    """回傳美金匯率歷史資料，供前端 Highcharts 畫圖。"""
    rates = UsdTwdRate.objects.order_by("rate_date")
    data = [
        {"date": r.rate_date.isoformat(), "close_rate": float(r.close_rate)}
        for r in rates
    ]
    return JsonResponse({"success": True, "message": "", "data": data})
