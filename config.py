# ******************************************************************************
# This contains all configs/parameters used in this project.
# ******************************************************************************

# Feature engineering
# ******************************************************************************

# filter co-events when doing self merge
MIN_TIME_TO_NEXT = 0  # value zero means that next event can't be before this event
MAX_TIME_TO_NEXT = 1 * 24 * 60 * 60  # 1 day * 24 hours * 60 min * 60 sec
MAX_TIME_TO_NEXT_CLICK_TO_CLICK = 3 * 60 * 60  # 3 hours * 60 min * 60 sec

OPTIM_ROWS_POLARS_GROUPBY = 100_000_000
MAX_ROWS_POLARS_GROUPBY = 350_000_000

# minimum count to be saved on disk
MIN_COUNT_TO_SAVE = {
    'count_click_to_click': 10,
    'count_click_to_cart_or_buy': 5,
    'count_cart_to_buy': 2,
    'count_cart_to_cart': 2,
    'count_buy_to_buy': 2,
}
MAX_CO_EVENT_PAIRS_TO_SAVE_DISK = 20_000_000

# which counts to compute
CO_EVENTS_TO_COUNT = [
    'count_click_to_click',
    'count_click_to_cart_or_buy',
    'count_cart_to_buy',
    'count_cart_to_cart',
    'count_buy_to_buy',
]
