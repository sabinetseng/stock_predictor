from django.http import JsonResponse
from .models import MarketIndexRate


def index_history(request, index_code):
    """回傳指數歷史資料（^TWII / ^SOX / ^VIX），供前端圖表使用。"""
    rates = MarketIndexRate.objects.filter(index_code=index_code).order_by("trade_date")
    data = [
        {"date": r.trade_date.isoformat(), "close_value": float(r.close_value)}
        for r in rates
    ]
    return JsonResponse({"success": True, "message": "", "data": data})
