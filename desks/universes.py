"""
Desk definitions for the multi-desk firm. Every ticker is a long/flat-tradeable
ETF; every desk runs the same four rules (multi-horizon trend, cross-sectional
12-1 momentum within the desk, 12% sleeve vol targeting, VIX/basket-vol crisis
liquidation), plus one desk-specific carry signal where a clean observable
exists:

  commodities -> futures-curve slope from front/deferred ETF pairs (energy)
  rates       -> yield-curve slope (^TNX 10y minus ^IRX 13-week): a steep
                 curve means positive roll-down carry for duration, an
                 inverted curve halves the TLT/IEF tilt
  equities/fx -> no clean carry observable via ETFs; trend + XS momentum only
                 (FX carry needs interest-rate differentials — future work)

Inception notes (staggered entry is supported by the engine — an asset joins
when its ETF launches): sector SPDRs 1998, EFA 2001, EEM 2003, VNQ 2004,
IEF/TLT/LQD 2002, TIP 2003, HYG/MBB 2007, EMB 2007, FXE 2005, FXB/FXF/FXA/FXC
2006, UUP/FXY 2007. SHY is deliberately excluded from rates: at ~1.5% vol it
duplicates the T-bill cash leg and, under inverse-vol sizing, would swallow
the whole sleeve.
"""

DATA_START = "2006-01-01"   # warmup for 200d SMA + vol estimators
EVAL_START = "2008-01-01"

DESKS = {
    "commodities": {
        "carry": "energy_curve",
        "sleeves": {
            "energy": {
                "WTI_12M": "USL", "BRENT": "BNO", "NG": "UNG", "GASOLINE": "UGA",
            },
            "agriculture": {
                "CORN": "CORN", "WHEAT": "WEAT", "SOY": "SOYB",
                "SUGAR": "CANE", "AG_BASKET": "DBA",
            },
            "metals": {
                "GOLD": "GLD", "SILVER": "SLV", "COPPER": "CPER",
                "PLATINUM": "PPLT", "PALLADIUM": "PALL", "BASE_METALS": "DBB",
            },
        },
    },
    "equities": {
        "sleeves": {
            "equities": {
                "TECH": "XLK", "FINANCIALS": "XLF", "ENERGY_EQ": "XLE",
                "HEALTHCARE": "XLV", "INDUSTRIALS": "XLI", "STAPLES": "XLP",
                "UTILITIES": "XLU", "DISCRETIONARY": "XLY", "MATERIALS": "XLB",
                "REITS": "VNQ", "INTL_DEV": "EFA", "INTL_EM": "EEM",
            },
        },
    },
    "rates": {
        "carry": "yield_curve",
        "sleeves": {
            "rates": {
                "TSY_7_10Y": "IEF", "TSY_20Y": "TLT", "TIPS": "TIP",
                "IG_CREDIT": "LQD", "HIGH_YIELD": "HYG",
                "MORTGAGES": "MBB", "EM_DEBT": "EMB",
            },
        },
    },
    "fx": {
        "sleeves": {
            "fx": {
                "USD": "UUP", "EUR": "FXE", "JPY": "FXY", "GBP": "FXB",
                "CHF": "FXF", "AUD": "FXA", "CAD": "FXC",
            },
        },
    },
}

# Assets the yield-curve carry tilt applies to (duration-sensitive Treasuries;
# credit/TIPS carry is driven by spreads, not the risk-free curve slope)
YIELD_CURVE_CARRY_ASSETS = ["TSY_7_10Y", "TSY_20Y"]
