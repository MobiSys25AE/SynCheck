import datetime

def get_current_time() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')