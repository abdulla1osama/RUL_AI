from sklearn.linear_model import Ridge


def build_linear_model(alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha)

    return model