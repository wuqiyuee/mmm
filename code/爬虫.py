import re
import time
import datetime as dt
from pathlib import Path
from collections import Counter

import pandas as pd
import requests
from zoneinfo import ZoneInfo


URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
TZ = ZoneInfo("Asia/Shanghai")

START_YEAR = 2000
END_YEAR = 2025
KEYWORDS = ["业绩预告", "业绩快报"]
# 巨潮网页端这个接口实际按每页 30 条分页。
# 如果改成 50，接口可能仍只返回 30 条，并且忽略 pageNum，导致第一页被重复保存。
PAGE_SIZE = 30
MAX_QUERY_PAGES = 100
MAX_RECORDS_PER_QUERY = PAGE_SIZE * MAX_QUERY_PAGES
SLEEP_SECONDS = 0.05

OUT_DIR = Path("巨潮业绩预告与业绩快报发布时间数据")
OUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
}


def strip_html(text):
    return re.sub(r"<[^>]+>", "", text or "")


def classify_time(timestamp_ms):
    t = dt.datetime.fromtimestamp(timestamp_ms / 1000, tz=TZ)
    tod = t.time()

    if tod == dt.time(0, 0):
        bucket = "时间未知(00:00:00)"
    elif dt.time(9, 30) <= tod <= dt.time(11, 30) or dt.time(13, 0) <= tod <= dt.time(15, 0):
        bucket = "盘中"
    elif dt.time(11, 30) < tod < dt.time(13, 0):
        bucket = "午间休市"
    elif tod < dt.time(9, 30):
        bucket = "盘前"
    else:
        bucket = "盘后"

    return t, bucket


def post_with_retry(session, data, max_retry=3):
    last_error = None

    for attempt in range(1, max_retry + 1):
        try:
            r = session.post(URL, headers=HEADERS, data=data, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
            time.sleep(0.8 * attempt)

    raise last_error


def announcement_key(announcement):
    return (
        announcement.get("announcementId"),
        announcement.get("secCode"),
        announcement.get("announcementTime"),
        strip_html(announcement.get("announcementTitle")),
    )


def split_date_range(start_date, end_date):
    if start_date >= end_date:
        raise RuntimeError(
            f"{start_date} 单日结果超过 {MAX_RECORDS_PER_QUERY} 条，"
            "已经无法继续按日期拆分。"
        )

    middle = start_date + (end_date - start_date) // 2
    return (start_date, middle), (middle + dt.timedelta(days=1), end_date)


def fetch_range(session, keyword, start_date, end_date):
    rows = []
    page_signatures = set()
    page = 1

    while True:
        data = {
            "pageNum": str(page),
            "pageSize": str(PAGE_SIZE),
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "",
            "searchkey": keyword,
            "secid": "",
            "category": "",
            "trade": "",
            "seDate": f"{start_date:%Y-%m-%d}~{end_date:%Y-%m-%d}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }

        j = post_with_retry(session, data)
        total = j.get("totalAnnouncement") or j.get("totalRecordNum") or 0
        anns = j.get("announcements") or []

        if page == 1 and total > MAX_RECORDS_PER_QUERY:
            left_range, right_range = split_date_range(start_date, end_date)
            left_rows = fetch_range(session, keyword, left_range[0], left_range[1])
            right_rows = fetch_range(session, keyword, right_range[0], right_range[1])
            return left_rows + right_rows

        if not anns:
            break

        page_signature = tuple(announcement_key(a) for a in anns)
        if page_signature in page_signatures:
            raise RuntimeError(
                f"{keyword} {start_date:%Y-%m-%d}~{end_date:%Y-%m-%d} "
                f"第 {page} 页和前面某一页完全相同。"
                f"请检查 PAGE_SIZE={PAGE_SIZE} 是否仍被巨潮接口支持。"
            )
        page_signatures.add(page_signature)

        rows.extend(anns)

        if len(rows) >= total:
            break

        page += 1
        time.sleep(SLEEP_SECONDS)

    return rows


def fetch_year(session, keyword, year):
    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)
    return fetch_range(session, keyword, start_date, end_date)


def main():
    detail_records = []
    summary_records = []

    with requests.Session() as session:
        for keyword in KEYWORDS:
            for year in range(START_YEAR, END_YEAR + 1):
                rows = fetch_year(session, keyword, year)
                counter = Counter()

                for a in rows:
                    title = strip_html(a.get("announcementTitle"))
                    timestamp_ms = a.get("announcementTime")

                    if keyword not in title or not timestamp_ms:
                        continue

                    publish_time, bucket = classify_time(timestamp_ms)
                    counter[bucket] += 1

                    detail_records.append({
                        "keyword": keyword,
                        "year": year,
                        "secCode": a.get("secCode"),
                        "secName": a.get("secName"),
                        "announcementTitle": title,
                        "announcementTime_ms": timestamp_ms,
                        "announcementTime": publish_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "timeBucket": bucket,
                        "adjunctUrl": a.get("adjunctUrl"),
                    })

                known_time = sum(
                    v for k, v in counter.items()
                    if k != "时间未知(00:00:00)"
                )

                summary_records.append({
                    "keyword": keyword,
                    "year": year,
                    "filtered": sum(counter.values()),
                    "known_time": known_time,
                    "unknown_000000": counter["时间未知(00:00:00)"],
                    "intraday": counter["盘中"],
                    "midday": counter["午间休市"],
                    "preopen": counter["盘前"],
                    "afterclose": counter["盘后"],
                })

                print(
                    f"{keyword} {year} 完成："
                    f"记录={sum(counter.values())}, 精确时刻={known_time}, "
                    f"盘中={counter['盘中']}, 盘后={counter['盘后']}"
                )

    detail_df = pd.DataFrame(detail_records)
    summary_df = pd.DataFrame(summary_records)

    detail_path = OUT_DIR / "cninfo_业绩预告_业绩快报_公告时间明细_2000_2025.csv"
    summary_path = OUT_DIR / "cninfo_业绩预告_业绩快报_公告时间汇总_2000_2025.csv"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n已保存：")
    print(detail_path.resolve())
    print(summary_path.resolve())

    print("\n总计：")
    print(
        summary_df
        .groupby("keyword")[["filtered", "known_time", "unknown_000000", "intraday", "midday", "preopen", "afterclose"]]
        .sum()
    )


if __name__ == "__main__":
    main()
