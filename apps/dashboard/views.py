from django.shortcuts import render


def index(request):
    """
    首頁視圖。
    TODO: 之後在這裡組合股票輸入框、Highcharts 圖表、
    預測按鈕、檢查新資料按鈕、檢查美金新資料按鈕所需的 context。
    """
    context = {}
    return render(request, "dashboard/index.html", context)
