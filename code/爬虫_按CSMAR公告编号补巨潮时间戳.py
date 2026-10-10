from __future__ import annotations

import argparse
import datetime as dt
import re
import time
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
CSMAR_PATH = ROOT / "中间结果/00_合并原始数据/CSMAR_公司公告事件表_2002_2025_合并结果.csv"
OUTPUT_PATH = ROOT / "巨潮业绩预告与业绩快报发布时间数据/csmar公告编号_巨潮时间戳匹配结果.csv"

CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STOCK_URL = "http://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_REFERER = "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
TZ = ZoneInfo("Asia/Shanghai")

PAGE_SIZE = 30
SLEEP_SECONDS = 0.08
REQUEST_TIMEOUT = 25

CSMAR_USECOLS = ["AnnouncementID", "Symbol", "DeclareDate", "Title"]
TITLE_PATTERN = r"业绩(?:预告|预报|快报)"
TITLE_EXCLUDE_PATTERN = (
    r"问询函|关注函|回复|专项说明|审核报告|核查意见|盈利预测|"
    r"业绩承诺|业绩补偿|承诺实现|实现情况|差异|监管函"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": CNINFO_REFERER,
    "Origin": "http://www.cninfo.com.cn",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def normalize_stock_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    match = re.search(r"\d{1,6}", text)
    if match is None:
        return ""
    return match.group(0).zfill(6)


def strip_html(text: object) -> str:
    return re.sub(r"<[^>]+>", "", "" if pd.isna(text) else str(text))


def extract_announcement_id(adjunct_url: object) -> str:
    text = "" if pd.isna(adjunct_url) else str(adjunct_url)
    match = re.search(r"/(\d+)\.(?:PDF|pdf|HTML|html)$", text)
    return match.group(1) if match else ""


def timestamp_to_local_time(timestamp_ms: object) -> str:
    if timestamp_ms in ("", None) or pd.isna(timestamp_ms):
        return ""
    timestamp = int(timestamp_ms)
    return dt.datetime.fromtimestamp(timestamp / 1000, tz=TZ).strftime("%Y-%m-%d %H:%M:%S")


def classify_time_bucket(local_time: object) -> str:
    if local_time in ("", None) or pd.isna(local_time):
        return "未匹配"
    value = pd.to_datetime(local_time, errors="coerce")
    if pd.isna(value):
        return "未匹配"

    tod = value.time()
    if tod == dt.time(0, 0):
        return "时间未知(00:00:00)"
    if dt.time(9, 30) <= tod <= dt.time(11, 30):
        return "盘中"
    if dt.time(11, 30) < tod < dt.time(13, 0):
        return "午间休市"
    if dt.time(13, 0) <= tod <= dt.time(15, 0):
        return "盘中"
    if tod < dt.time(9, 30):
        return "盘前"
    return "盘后"


def post_with_retry(session: requests.Session, data: dict[str, str], max_retry: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, max_retry + 1):
        try:
            response = session.post(
                CNINFO_QUERY_URL,
                headers=HEADERS,
                data=data,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"巨潮请求失败：{last_error}") from last_error


def load_cninfo_stock_map(session: requests.Session) -> dict[str, list[str]]:
    response = session.get(CNINFO_STOCK_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    stock_map: dict[str, list[str]] = {}

    for item in data.get("stockList", []):
        code = normalize_stock_code(item.get("code"))
        org_id = str(item.get("orgId") or "").strip()
        if code and org_id:
            stock_map.setdefault(code, []).append(org_id)

    return stock_map


def build_stock_params(stock_code: str, stock_map: dict[str, list[str]]) -> list[str]:
    org_ids = stock_map.get(stock_code, [])
    params = [f"{stock_code},{org_id}" for org_id in org_ids]
    if params:
        return params

    if stock_code.startswith(("000", "001", "002", "003", "200", "300")):
        return [f"{stock_code},gssz{stock_code}"]
    if stock_code.startswith(("600", "601", "603", "605", "688", "900")):
        return [f"{stock_code},ggsh{stock_code}"]
    if stock_code.startswith(("430", "800", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "920")):
        return [f"{stock_code},gfbj{stock_code}"]
    return []


def make_query_data(stock_param: str, declare_date: str, page_num: int) -> dict[str, str]:
    return {
        "pageNum": str(page_num),
        "pageSize": str(PAGE_SIZE),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": stock_param,
        "searchkey": "",
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{declare_date}~{declare_date}",
        "sortName": "time",
        "sortType": "desc",
        "isHLtitle": "true",
    }


def fetch_cninfo_announcements(
    session: requests.Session,
    stock_param: str,
    declare_date: str,
) -> list[dict]:
    rows: list[dict] = []
    page_num = 1

    while True:
        data = make_query_data(stock_param, declare_date, page_num)
        result = post_with_retry(session, data)
        announcements = result.get("announcements") or []
        total = int(result.get("totalAnnouncement") or result.get("totalRecordNum") or 0)

        if not announcements:
            break

        rows.extend(announcements)
        if len(rows) >= total:
            break
        page_num += 1
        time.sleep(SLEEP_SECONDS)

    return rows


def read_csmar_targets(
    csmar_path: Path,
    start_date: str | None,
    end_date: str | None,
    limit: int | None,
    chunksize: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    total_kept = 0
    start = pd.to_datetime(start_date) if start_date else None
    end = pd.to_datetime(end_date) if end_date else None

    reader = pd.read_csv(
        csmar_path,
        encoding="utf-8-sig",
        usecols=CSMAR_USECOLS,
        dtype={"AnnouncementID": "string", "Symbol": "string", "Title": "string"},
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in reader:
        chunk = chunk.copy()
        chunk["AnnouncementID"] = chunk["AnnouncementID"].astype("string").str.strip()
        chunk["Stkcd"] = chunk["Symbol"].apply(normalize_stock_code)
        chunk["DeclareDate"] = pd.to_datetime(chunk["DeclareDate"], errors="coerce")

        title = chunk["Title"].fillna("").astype(str)
        mask = (
            chunk["AnnouncementID"].notna()
            & chunk["Stkcd"].ne("")
            & chunk["DeclareDate"].notna()
            & title.str.contains(TITLE_PATTERN, regex=True)
            & ~title.str.contains(TITLE_EXCLUDE_PATTERN, regex=True)
        )
        if start is not None:
            mask &= chunk["DeclareDate"].ge(start)
        if end is not None:
            mask &= chunk["DeclareDate"].le(end)

        kept = chunk.loc[mask, ["AnnouncementID", "Stkcd", "DeclareDate", "Title"]].copy()
        if kept.empty:
            continue

        kept["DeclareDate"] = kept["DeclareDate"].dt.strftime("%Y-%m-%d")
        parts.append(kept)
        total_kept += len(kept)

        if limit is not None and total_kept >= limit:
            break

    if not parts:
        return pd.DataFrame(columns=["AnnouncementID", "Stkcd", "DeclareDate", "Title"])

    targets = pd.concat(parts, ignore_index=True)
    targets = targets.drop_duplicates(["AnnouncementID", "Stkcd", "DeclareDate", "Title"])
    targets = targets.sort_values(["DeclareDate", "Stkcd", "AnnouncementID"]).reset_index(drop=True)
    if limit is not None:
        targets = targets.head(limit).copy()
    return targets


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = pd.read_csv(
        output_path,
        encoding="utf-8-sig",
        usecols=["AnnouncementID"],
        dtype={"AnnouncementID": "string"},
        low_memory=False,
    )
    return set(done["AnnouncementID"].dropna().astype(str))


def classify_target_keyword(title: str) -> str:
    has_forecast = bool(re.search(r"业绩(?:预告|预报)", title))
    has_brief = "业绩快报" in title
    if has_forecast and has_brief:
        return "业绩预告;业绩快报"
    if has_forecast:
        return "业绩预告"
    if has_brief:
        return "业绩快报"
    return ""


def match_one_target(
    session: requests.Session,
    target: pd.Series,
    stock_map: dict[str, list[str]],
) -> dict[str, object]:
    announcement_id = str(target["AnnouncementID"])
    stock_code = str(target["Stkcd"])
    declare_date = str(target["DeclareDate"])
    title = str(target["Title"])

    stock_params = build_stock_params(stock_code, stock_map)
    if not stock_params:
        return {
            "AnnouncementID": announcement_id,
            "Stkcd": stock_code,
            "DeclareDate": declare_date,
            "Title": title,
            "keyword": classify_target_keyword(title),
            "match_status": "no_stock_org_id",
        }

    checked_rows = 0
    for stock_param in stock_params:
        announcements = fetch_cninfo_announcements(session, stock_param, declare_date)
        checked_rows += len(announcements)

        for item in announcements:
            cninfo_id = str(item.get("announcementId") or extract_announcement_id(item.get("adjunctUrl")))
            if cninfo_id != announcement_id:
                continue

            announcement_time_ms = item.get("announcementTime")
            local_time = timestamp_to_local_time(announcement_time_ms)
            return {
                "AnnouncementID": announcement_id,
                "Stkcd": stock_code,
                "DeclareDate": declare_date,
                "Title": title,
                "keyword": classify_target_keyword(title),
                "match_status": "matched",
                "cninfo_secCode": normalize_stock_code(item.get("secCode")),
                "cninfo_secName": strip_html(item.get("secName")),
                "cninfo_title": strip_html(item.get("announcementTitle")),
                "cninfo_announcementTime_ms": announcement_time_ms,
                "cninfo_announcementTime": local_time,
                "cninfo_timeBucket": classify_time_bucket(local_time),
                "cninfo_adjunctUrl": item.get("adjunctUrl"),
                "cninfo_checked_rows": checked_rows,
                "stock_param": stock_param,
            }

        time.sleep(SLEEP_SECONDS)

    return {
        "AnnouncementID": announcement_id,
        "Stkcd": stock_code,
        "DeclareDate": declare_date,
        "Title": title,
        "keyword": classify_target_keyword(title),
        "match_status": "not_found_on_cninfo_day_stock",
        "cninfo_checked_rows": checked_rows,
        "stock_param": ";".join(stock_params),
    }


def append_results(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=header, index=False, encoding="utf-8-sig")


def iter_batches(values: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(values), batch_size):
        yield values.iloc[start : start + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="依据 CSMAR 公告编号去巨潮补充公告时间戳")
    parser.add_argument("--csmar", type=Path, default=CSMAR_PATH, help="CSMAR 公司公告事件表路径")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="输出 CSV 路径")
    parser.add_argument("--start-date", type=str, default=None, help="起始公告日期，例如 2021-01-01")
    parser.add_argument("--end-date", type=str, default=None, help="结束公告日期，例如 2021-12-31")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条，适合测试")
    parser.add_argument("--chunksize", type=int, default=500_000, help="读取 CSMAR 的分块大小")
    parser.add_argument("--batch-size", type=int, default=100, help="每处理多少条追加保存一次")
    parser.add_argument("--resume", action="store_true", help="如果输出文件已存在，跳过已处理 AnnouncementID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = read_csmar_targets(
        csmar_path=args.csmar,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        chunksize=args.chunksize,
    )

    if args.resume:
        done_ids = load_done_ids(args.output)
        targets = targets.loc[~targets["AnnouncementID"].astype(str).isin(done_ids)].copy()

    print(f"待匹配 CSMAR 公告数：{len(targets)}")
    if targets.empty:
        return

    with requests.Session() as session:
        print("读取巨潮股票 orgId 映射...")
        stock_map = load_cninfo_stock_map(session)
        print(f"股票映射数量：{len(stock_map)}")

        processed = 0
        for batch in iter_batches(targets, args.batch_size):
            results = []
            for _, target in batch.iterrows():
                result = match_one_target(session, target, stock_map)
                results.append(result)
                processed += 1
                if processed % 10 == 0:
                    print(
                        f"已处理 {processed}/{len(targets)}，"
                        f"最近状态={result.get('match_status')}",
                        flush=True,
                    )
                time.sleep(SLEEP_SECONDS)

            append_results(args.output, results)

    print("=" * 80)
    print("匹配完成")
    print("输出文件：", args.output)


if __name__ == "__main__":
    main()
