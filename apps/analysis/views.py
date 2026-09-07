from django.http import JsonResponse, HttpResponseRedirect

from . import services


def correlation_page(request):
    """
    相關性分析已整併進個股「決策看板」頁（/stocks/<代碼>/ 底部的相關性分析面板）。
    本路由僅保留舊連結相容性：
    - /analysis/?stock_code=2330 → 轉址到 /stocks/2330/#correlation-panel
    - /analysis/                 → 轉址回首頁
    JSON API（/analysis/correlation/<stock_code>/）不受影響，決策看板仍呼叫它。
    """
    stock_code = request.GET.get("stock_code", "").strip()
    if stock_code:
        return HttpResponseRedirect(f"/stocks/{stock_code}/#correlation-panel")
    return HttpResponseRedirect("/")


def correlation(request, stock_code):
    """
    回傳個股與美金匯率 / 大盤的關聯分析結果，供前端圖表顯示。
    query string 參數：
        compare_target: USD_TWD / TWII / SOX / VIX（預設 USD_TWD）
    """
    compare_target = request.GET.get("compare_target", "USD_TWD")
    result = services.calc_all_correlations(stock_code, compare_target)

    overall_success = all(r["success"] for r in result.values())
    return JsonResponse({
        "success": overall_success,
        "message": "計算完成" if overall_success else "部分分析失敗，詳見 data 內容",
        "data": result,
    })
