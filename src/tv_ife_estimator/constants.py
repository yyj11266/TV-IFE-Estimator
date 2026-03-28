"""Project-wide constants."""

from __future__ import annotations

RAW_SHEET_NAME = "宝贝搜索-路由器-20240719134137_1"
ENHANCED_SHEET_NAME = "数据源"

RAW_FILENAME = "router_market_raw.xlsx"
ENHANCED_FILENAME = "router_market_enhanced.xlsx"

ALL_MONTHS = (
    "202307",
    "202308",
    "202309",
    "202310",
    "202311",
    "202312",
    "202401",
    "202402",
    "202403",
    "202404",
    "202405",
    "202406",
    "202407",
)

FULL_MONTHS = ALL_MONTHS[:-1]
DEFAULT_TARGET_MONTHS = ("202403", "202404", "202405", "202406")

WIFI_GENERATION_MAP = {
    "WiFi4": 4,
    "WiFi5": 5,
    "WiFi6": 6,
    "WiFi7": 7,
}

STATIC_NUMERIC_FIELDS = (
    "参考价格",
    "传输速率\nMbps",
    "WAN/LAN口数量",
    "2.5G网口数",
    "内存",
    "FEM",
    "最高频宽\nMHz",
)

STATIC_CATEGORICAL_FIELDS = (
    "网络协议",
    "mesh组网",
)

CORE_DIRECT_NUMERIC_FEATURES = (
    "log_prev_sales",
    "log_prev_prev_sales",
    "prev_sales_growth",
    "log_prev_two_month_mean",
    "prev_asp",
    "prev_discount",
    "wifi_gen",
    "log_speed_mbps",
    "mesh_flag",
    "wan_lan_ports",
    "log_ref_price",
)

CORE_DIRECT_CATEGORICAL_FEATURES = ("brand",)

CORE_PAPER_NUMERIC_FEATURES = (
    "current_log_asp",
    "current_discount",
)
