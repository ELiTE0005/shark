#!/usr/bin/env python3
"""
Enhanced Etherscan wallet analyzer + plot.

Features added:
- argparse for CLI usage
- API key from env var (fallback to your constant)
- error handling & retries (simple)
- support for normal txns + internal txns + token transfers (ERC20/ERC721)
- summary stats: total txns, incoming/outgoing counts, total fees, largest txs
- saves CSV of transactions
- nicer matplotlib figure with:
    * balance over time (line)
    * tx value bars (incoming/outgoing)
    * cumulative fees (secondary subplot)
    * rolling average of balance
- ability to save plot to file
"""

import os
import time
import csv
import argparse
from datetime import datetime
from collections import defaultdict, namedtuple

import requests
import matplotlib.dates as mdates
from matplotlib import pyplot as plt

# ---------- CONFIG ----------
API_KEY = os.getenv("ETHERSCAN_API_KEY", "DI8CZVGXKHRQ1MF2UJ6Q2WH77KP12HDSHM")
BASE_URL = "https://api.etherscan.io/api"
ETHER_VALUE = 10 ** 18
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 1.2
# ----------------------------

TxRecord = namedtuple("TxRecord", [
    "time", "hash", "from_addr", "to_addr", "value_eth",
    "gas_used", "gas_price", "fee_eth", "is_internal", "is_token", "token_symbol"
])

def make_api_url(module, action, address, **kwargs):
    url = BASE_URL + f"?module={module}&action={action}&address={address}&apikey={API_KEY}"
    for key, value in kwargs.items():
        url += f"&{key}={value}"
    return url

def safe_get(url, params=None, retries=REQUEST_RETRIES):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "0" and "message" in data:
                # Etherscan sometimes returns status 0 for "No transactions found"
                # Return data anyway to let caller handle it
                return data
            return data
        except requests.RequestException as e:
            if attempt + 1 == retries:
                raise
            time.sleep((REQUEST_BACKOFF ** attempt))
    raise RuntimeError("request failed after retries")

def parse_tx_item(tx, is_internal=False, is_token=False):
    """Normalise a tx/item from various API endpoints into TxRecord"""
    # many token endpoints use different fields; handle common cases
    timestamp = int(tx.get("timeStamp") or tx.get("timeStamp") or tx.get("blockNumber") or 0)
    try:
        time_dt = datetime.fromtimestamp(int(timestamp))
    except Exception:
        time_dt = datetime.utcnow()

    value = int(tx.get("value", 0))
    value_eth = value / ETHER_VALUE

    gas_used = int(tx.get("gasUsed") or tx.get("gas") or 0)
    gas_price = int(tx.get("gasPrice") or tx.get("gasPriceGwei") or 0)
    fee_eth = (gas_used * gas_price) / ETHER_VALUE if (gas_used and gas_price) else 0.0

    token_symbol = tx.get("tokenSymbol") if is_token else None

    return TxRecord(
        time=time_dt,
        hash=tx.get("hash") or tx.get("transactionHash"),
        from_addr=(tx.get("from") or "").lower(),
        to_addr=(tx.get("to") or "").lower(),
        value_eth=value_eth,
        gas_used=gas_used,
        gas_price=gas_price,
        fee_eth=fee_eth,
        is_internal=is_internal,
        is_token=is_token,
        token_symbol=token_symbol
    )

def fetch_all_transactions(address, include_tokens=False, page_size=10000):
    address = address.lower()
    # normal tx list
    txs = []
    url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=page_size, sort="asc")
    data = safe_get(url)
    results = data.get("result") or []
    txs.extend([parse_tx_item(t, is_internal=False, is_token=False) for t in results])

    # internal txs
    url2 = make_api_url("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=page_size, sort="asc")
    data2 = safe_get(url2)
    results2 = data2.get("result") or []
    txs.extend([parse_tx_item(t, is_internal=True, is_token=False) for t in results2])

    # token transfers (ERC20)
    token_txs = []
    if include_tokens:
        url3 = make_api_url("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=page_size, sort="asc")
        data3 = safe_get(url3)
        results3 = data3.get("result") or []
        token_txs.extend([parse_tx_item(t, is_internal=False, is_token=True) for t in results3])

        # NFT transfers (optional)
        url4 = make_api_url("account", "tokennfttx", address, startblock=0, endblock=99999999, page=1, offset=page_size, sort="asc")
        data4 = safe_get(url4)
        results4 = data4.get("result") or []
        token_txs.extend([parse_tx_item(t, is_internal=False, is_token=True) for t in results4])

    # combine and sort
    txs.extend(token_txs)
    txs = [t for t in txs if t.time is not None]
    txs.sort(key=lambda x: x.time)
    return txs

def compute_balance_series(txs, address):
    """Given chronological txs, compute running balance (ETH) and produce series for plotting"""
    addr = address.lower()
    balance = 0.0
    balances = []
    times = []
    values = []  # positive for incoming, negative for outgoing (value only)
    fees = []
    incoming_count = 0
    outgoing_count = 0
    total_fees = 0.0

    for tx in txs:
        # treat token txs separately (we won't include token value in ETH balance)
        if tx.is_token:
            # skip token value affecting ETH balance; still can record it if desired
            balances.append(balance)
            times.append(tx.time)
            values.append(0.0)
            fees.append(tx.fee_eth)
            total_fees += tx.fee_eth
            continue

        money_in = (tx.to_addr == addr)
        if money_in:
            balance += tx.value_eth
            incoming_count += 1
            values.append(tx.value_eth)
        else:
            # outgoing: subtract value and fee
            balance -= (tx.value_eth + tx.fee_eth)
            outgoing_count += 1
            values.append(-(tx.value_eth + tx.fee_eth))

        total_fees += tx.fee_eth
        fees.append(total_fees)  # cumulative fees at this point
        balances.append(balance)
        times.append(tx.time)

    return {
        "times": times,
        "balances": balances,
        "values": values,
        "fees_cumulative": fees,
        "incoming_count": incoming_count,
        "outgoing_count": outgoing_count,
        "total_fees": total_fees
    }

def save_csv(txs, filename):
    fieldnames = ["timestamp", "hash", "from", "to", "value_eth", "fee_eth", "is_internal", "is_token", "token_symbol"]
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in txs:
            writer.writerow({
                "timestamp": t.time.isoformat(),
                "hash": t.hash,
                "from": t.from_addr,
                "to": t.to_addr,
                "value_eth": t.value_eth,
                "fee_eth": t.fee_eth,
                "is_internal": t.is_internal,
                "is_token": t.is_token,
                "token_symbol": t.token_symbol or ""
            })

def plot_results(series, txs, address, save_path=None, rolling_window=20):
    times = series["times"]
    balances = series["balances"]
    values = series["values"]
    fees_cum = series["fees_cumulative"]

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[3,1]})

    # balance line + rolling avg
    ax1.plot(times, balances, label="Balance (ETH)", linewidth=1.8)
    if len(balances) >= 3:
        # simple rolling average
        import numpy as np
        roll = np.convolve(balances, np.ones(min(len(balances), rolling_window))/min(len(balances), rolling_window), mode='valid')
        # align roll with times
        ax1.plot(times[len(times)-len(roll):], roll, linestyle='--', label=f"{rolling_window}-tx rolling avg")

    ax1.set_ylabel("Balance (ETH)")
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left')

    # bar chart for values (incoming/outgoing)
    pos_vals = [v if v>0 else 0 for v in values]
    neg_vals = [v if v<0 else 0 for v in values]
    ax1.bar(times, pos_vals, width=0.01, alpha=0.4, label="Incoming (ETH)")
    ax1.bar(times, neg_vals, width=0.01, alpha=0.4, label="Outgoing+Fee (ETH)")
    ax1.legend(loc='upper right')

    # cumulative fees subplot
    if fees_cum:
        ax2.plot(times, fees_cum, label="Cumulative Fees (ETH)", linewidth=1)
        ax2.set_ylabel("Fees (ETH)")
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend()

    # formatting x-axis
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=25)
    plt.suptitle(f"Address: {address}\nTotal txns: {len(txs)}   Incoming: {series['incoming_count']}   Outgoing: {series['outgoing_count']}   Total fees: {series['total_fees']:.6f} ETH")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")

    plt.show()

def summarize_top_transactions(txs, top_n=5):
    # show top by absolute ETH moved (ignoring fees)
    txs_eth = [t for t in txs if not t.is_token]
    txs_sorted = sorted(txs_eth, key=lambda x: abs(x.value_eth), reverse=True)
    print(f"\nTop {top_n} transactions by ETH value (abs):")
    for t in txs_sorted[:top_n]:
        direction = "IN " if t.value_eth > 0 else "OUT"
        print(f"- {t.time.isoformat()} | {direction} {abs(t.value_eth):.6f} ETH | fee {t.fee_eth:.6f} ETH | hash {t.hash}")

def main():
    parser = argparse.ArgumentParser(description="Etherscan wallet transaction visualizer")
    parser.add_argument("address", help="Ethereum address to analyze")
    parser.add_argument("--tokens", action="store_true", help="Include token/ERC721 transfers (won't affect ETH balance)")
    parser.add_argument("--save-plot", help="Path to save plot image (png)")
    parser.add_argument("--save-csv", help="Path to save transactions CSV")
    parser.add_argument("--no-show", action="store_true", help="Don't show interactive plot (useful when only saving)")
    args = parser.parse_args()

    address = args.address
    print(f"Fetching txs for {address} (include tokens: {args.tokens}) ...")
    txs = fetch_all_transactions(address, include_tokens=args.tokens)
    print(f"Total returned items: {len(txs)}")

    series = compute_balance_series(txs, address)
    print(f"Incoming txns: {series['incoming_count']}, Outgoing txns: {series['outgoing_count']}, total fees: {series['total_fees']:.6f} ETH")

    # save CSV if requested
    if args.save_csv:
        save_csv(txs, args.save_csv)
        print(f"Saved {len(txs)} transactions to {args.save_csv}")

    # summary top transactions
    summarize_top_transactions(txs, top_n=6)

    # plot
    if not args.no_show:
        plot_results(series, txs, address, save_path=args.save_plot)
    else:
        if args.save_plot:
            # still save but don't show
            plot_results(series, txs, address, save_path=args.save_plot)
        else:
            print("no-show and no save-plot specified; nothing to display.")

if __name__ == "__main__":
    main()


