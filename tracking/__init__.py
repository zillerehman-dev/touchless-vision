def smooth(previous, current, keep):
    # Exponential moving average: keep * old + (1 - keep) * new.
    # Works for a single number or a tuple of numbers; None means "no history yet".
    if previous is None:
        return current
    if isinstance(current, (int, float)):
        return keep * previous + (1 - keep) * current
    return tuple(keep * p + (1 - keep) * c for p, c in zip(previous, current))
