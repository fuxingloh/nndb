#!/usr/bin/env python3
"""Assemble the hardware cost-right-sizing sweep (history 062) from the per-box
JSON in history/hw/*.jsonl + /tmp/hw-pricing.tsv. Recall is held constant across
boxes by construction, so the axes that matter are QPS, p50, and QPS-per-dollar.

Usage: python3 history/hw-report.py            # prints markdown table + JSON blob
"""
import json, glob, os, sys

HW = os.path.join(os.path.dirname(__file__), "hw")
PRICING = "/tmp/hw-pricing.tsv"

# physical cores + ISA per instance (from lscpu in the run logs) — the key confound:
# "8 vCPU" = 4 Intel cores + SMT, vs 8 real cores on AMD/Graviton (no SMT).
META = {
    "c8i.2xlarge": ("Intel Granite Rapids", 4, "AVX-512 VPOPCNTDQ", "compute"),
    "m8i.2xlarge": ("Intel Granite Rapids", 4, "AVX-512 VPOPCNTDQ", "general"),
    "c8a.2xlarge": ("AMD Zen5 (Turin)",     8, "AVX-512 VPOPCNTDQ", "compute"),
    "m8a.2xlarge": ("AMD Zen5 (Turin)",     8, "AVX-512 VPOPCNTDQ", "general"),
    "c8g.2xlarge": ("Graviton4",            8, "NEON CNT",          "compute"),
    "m8g.2xlarge": ("Graviton4",            8, "NEON CNT",          "general"),
    "mac2.metal":  ("Apple M1",             8, "NEON CNT",          "mac"),
}

def load_pricing():
    p = {}
    with open(PRICING) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split("\t")
            p[c[0]] = {"od": float(c[1]), "spot": (None if c[2] == "NA" else float(c[2]))}
    return p

def load_box(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        kind = "funnel" if o["quant"] == "binary" else "exact"
        rows[kind] = o
    return rows

def main():
    pricing = load_pricing()
    boxes = {}
    for f in sorted(glob.glob(os.path.join(HW, "*.jsonl"))):
        name = os.path.basename(f)[:-6]
        r = load_box(f)
        if "funnel" in r:
            boxes[name] = r

    VCPU = 8  # every box is the advised .2xlarge = 8 vCPU
    HRS_YR = 24 * 365
    table = []
    for name, r in boxes.items():
        fun, ex = r["funnel"], r.get("exact", {})
        vendor, cores, isa, fam = META.get(name, ("?", 0, "?", "?"))
        od = pricing.get(name, {}).get("od")
        spot = pricing.get(name, {}).get("spot")
        qps = fun["qps"]
        usd_yr = round(od * HRS_YR) if od else None
        table.append({
            "instance": name, "vendor": vendor, "family": fam,
            "vcpu": VCPU, "phys_cores": cores, "isa": isa,
            "funnel_qps": qps, "funnel_p50_ms": fun["latency_ms"]["p50"],
            "recall": fun["recall_at_k"],
            "od_usd_yr": usd_yr, "spot_usd_hr": spot,
            "qps_per_1k_usd_yr": round(qps / (usd_yr / 1000), 0) if usd_yr else None,
            "qps_per_vcpu": round(qps / VCPU, 0),
        })
    table.sort(key=lambda x: -(x["qps_per_1k_usd_yr"] or 0))

    # markdown
    print("| instance (8 vCPU) | vendor | vCPU | phys | popcount | funnel QPS | p50 ms | recall | $/yr (od) | QPS per $1k/yr | QPS/vCPU |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in table:
        print(f"| {t['instance']} | {t['vendor']} | {t['vcpu']} | {t['phys_cores']} | {t['isa']} | "
              f"{t['funnel_qps']:.0f} | {t['funnel_p50_ms']:.1f} | {t['recall']:.4f} | "
              f"${t['od_usd_yr']:,} | {t['qps_per_1k_usd_yr']:.0f} | {t['qps_per_vcpu']:.0f} |")
    print("\n--- JSON ---")
    print(json.dumps(table, indent=2))

if __name__ == "__main__":
    main()
