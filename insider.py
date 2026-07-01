"""
--------------------------------------------
INSIDER FLOW COLLECTOR (SEC EDGAR Form 4)
--------------------------------------------

Edge #1: legal, public insider-transaction data from SEC EDGAR (free).
Maps ticker -> CIK, pulls Form 4 filings, parses open-market buys/sells, and
(later) aggregates to a daily net-insider-buying signal per ticker.

Point-in-time: a Form 4 filed on date D is only known at/after D, so the signal
for day D must use filings with filingDate <= D (no look-ahead).

SEC rules honored: <=10 req/sec, User-Agent with a contact email (else 403).
The strongest signal (per research) is a CLUSTER BUY - 3+ insiders buying on the
open market (code 'P') in a short window.
"""

import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

CONTACT = "mscibak997@gmail.com"
HEADERS = {"User-Agent": f"Stock_trade research {CONTACT}"}
RATE_SLEEP = 0.15                       # stay well under 10 req/sec

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"


def _get(url: str):
    time.sleep(RATE_SLEEP)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def _local(tag: str) -> str:
    # strip any XML namespace -> local tag name
    return tag.rsplit("}", 1)[-1]


def _field_value(el) -> str:
    # transactionCode carries direct text; shares/price/AD wrap it in a <value> child
    for child in el:
        if _local(child.tag) == "value":
            return (child.text or "").strip()
    return (el.text or "").strip()


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_cik_map() -> dict:
    data = _get(TICKERS_URL).json()
    return {row["ticker"]: int(row["cik_str"]) for row in data.values()}


def get_form4_filings(cik: int) -> pd.DataFrame:
    recent = _get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
    df = pd.DataFrame({"form": recent["form"],
                       "filingDate": recent["filingDate"],
                       "accession": recent["accessionNumber"]})
    return df[df["form"] == "4"].reset_index(drop=True)


def parse_form4_xml(xml_text: str) -> list[dict]:
    # non-derivative transactions: code (P=buy, S=sell), A/D, shares, price
    root = ET.fromstring(xml_text)
    txns = []
    for el in root.iter():
        if _local(el.tag) != "nonDerivativeTransaction":
            continue
        fields = {}
        for sub in el.iter():
            name = _local(sub.tag)
            if name in ("transactionCode", "transactionShares",
                        "transactionPricePerShare", "transactionAcquiredDisposedCode"):
                fields[name] = _field_value(sub)
        if fields.get("transactionCode"):
            txns.append({
                "code": fields["transactionCode"],
                "ad": fields.get("transactionAcquiredDisposedCode"),
                "shares": _to_float(fields.get("transactionShares")),
                "price": _to_float(fields.get("transactionPricePerShare")),
            })
    return txns


def parse_form4(cik: int, accession: str) -> list[dict]:
    # fetch a filing's ownership XML from EDGAR, then parse it
    acc = accession.replace("-", "")
    idx = _get(f"{ARCHIVE.format(cik=cik, acc=acc)}/index.json").json()
    xml_name = next((f["name"] for f in idx["directory"]["item"] if f["name"].endswith(".xml")), None)
    if not xml_name:
        return []
    return parse_form4_xml(_get(f"{ARCHIVE.format(cik=cik, acc=acc)}/{xml_name}").text)


def _demo(ticker: str = "AAPL", n: int = 6) -> None:
    print(f"\nInsider flow (SEC EDGAR Form 4) - {ticker} :\n")
    cik = load_cik_map()[ticker]
    print(f"  CIK           : {cik}")
    f4 = get_form4_filings(cik)
    print(f"  Form 4 filings: {len(f4)} recent  (latest {f4['filingDate'].iloc[0]})\n")
    for _, row in f4.head(n).iterrows():
        txns = parse_form4(cik, row["accession"])
        buys = [t for t in txns if t["code"] == "P"]
        sells = [t for t in txns if t["code"] == "S"]
        val = lambda ts: sum(t["shares"] * t["price"] for t in ts)
        print(f"  {row['filingDate']}  P(buy) {len(buys):>2} ~${val(buys):>12,.0f}   "
              f"S(sell) {len(sells):>2} ~${val(sells):>13,.0f}")


if __name__ == "__main__":
    _demo()
