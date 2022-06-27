def PerformanceThresholdOp(
    json_url: str,
    f1_threshold: float,
) -> str:
    import pandas as pd
    
    results = pd.read_json(json_url, typ='series')
    if results['FullModel'] >= f1_threshold:
        return 'pass'
    else:
        return 'fail'