import datetime

def get_current_time() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def scaler(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    getMax = x.max()
    x = ((x - x.min())/(getMax - x.min()))
    
    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x