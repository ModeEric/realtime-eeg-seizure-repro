def epoch_metric(y_true, y_pred, thresh=0.5):
    return ( (y_true == (y_pred > thresh)).mean() )