from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import requests
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

app = FastAPI(title="Currency Peak Analyzer")

JAVA_BACKEND_URL = "http://localhost:8080/currency"

NEWS_API_KEY = "23c60730daa94dadb568a7c9ceea0672"


class Currency(BaseModel):
    date: str
    code: str
    mid: Decimal
    internetAnswer: str


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def fill_missing_days(data):
    """
    Wypełnia brakujące dni średnią z sąsiednich kursów.
    Zakładamy, że dane są posortowane po dacie rosnąco.
    """
    if not data:
        return []

    filled_data = [data[0]]

    for i in range(1, len(data)):
        prev_date = parse_date(filled_data[-1]["date"])
        curr_date = parse_date(data[i]["date"])
        prev_mid = Decimal(filled_data[-1]["mid"])
        curr_mid = Decimal(data[i]["mid"])
        gap_days = (curr_date - prev_date).days - 1
        if gap_days > 0:
            step = (curr_mid - prev_mid) / (gap_days + 1)
            for j in range(1, gap_days + 1):
                filled_mid = prev_mid + step * j
                filled_date = (prev_date + timedelta(days=j)).strftime("%Y-%m-%d")
                filled_data.append({"date": filled_date, "mid": float(filled_mid)})

        filled_data.append(data[i])

    return filled_data


def detect_peaks(data, window: int = 3, k: float = 3.0, min_diff: float = 0.2):
    """
    Wykrywa anomalie: duże odchylenia od lokalnej średniej
    sprawdzane względem sigma ORAZ minimalnej różnicy w złotówkach.
    """
    values = np.array([d["mid"] for d in data])
    n = len(values)
    peaks = []
    for i, val in enumerate(values):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        neighbors = [values[j] for j in range(start, end) if j != i]
        if not neighbors:
            continue

        local_avg = np.mean(neighbors)
        local_std = np.std(neighbors)

        diff = abs(val - local_avg)
        if local_std < 1e-6:
            continue

        z_score = diff / local_std

        if z_score >= k and diff >= min_diff:
            peaks.append(data[i])
    return peaks


def search_internet(code:str,query: str, real_date: str, from_date: str = None, to_date: str = None, max_results: int = 3) -> str:
    """
    Wyszukuje artykuły online za pomocą NewsAPI.
    Zwraca tytuł i źródło pierwszego artykułu albo pusty string.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results,
        "apiKey": NEWS_API_KEY
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return f"Error fetching news: {resp.status_code}"

    data = resp.json().get("articles", [])
    if not data:
        return f"No relevant news found for {code} around {real_date}."
    article = data[0]
    title = article.get("title", "No title")
    source = article.get("source", {}).get("name", "")
    return f"{title} ({source})"


@app.get("/currency", response_model=list[Currency])
def analyze_currency(
        code: str = Query(..., description="Kod waluty np. USD"),
        start_date: str = Query(..., description="Data początkowa yyyy-MM-dd"),
        end_date: str = Query(..., description="Data końcowa yyyy-MM-dd")
):
    try:
        resp = requests.get(
            f"{JAVA_BACKEND_URL}/{code}?startDate={start_date}&endDate={end_date}"
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return []
        data_filled = fill_missing_days(data)
        peaks = detect_peaks(data_filled)
        peak_dates = {p["date"] for p in peaks}
        result = []
        for day in data_filled:
            internet_info = ""
            if day["date"] in peak_dates:
                date_obj = datetime.strptime(day["date"], "%Y-%m-%d")
                from_date = (date_obj - timedelta(days=3)).strftime("%Y-%m-%d")
                to_date = (date_obj + timedelta(days=3)).strftime("%Y-%m-%d")
                internet_info = search_internet(
                    code=code,
                    query=f"{code} exchange rate",
                    real_date=day["date"],
                    from_date=from_date,
                    to_date=to_date
                )
            result.append(
                Currency(
                    date=day["date"],
                    code=code,
                    mid=day["mid"],
                    internetAnswer=internet_info
                )
            )
        return result

    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Błąd połączenia z backendem Java: {str(e)}")
