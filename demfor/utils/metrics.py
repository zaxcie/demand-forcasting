from keras import backend as K


def smape(preds, target):
    '''
    Function to calculate SMAPE
    '''
    smape_val = 0
    for pred,true in zip(preds, target):
        if (pred == 0) & (true == 0):
            continue
        else:
            smape_val += abs(pred-true)/(abs(pred)+abs(true))
    smape_val=(200*smape_val)/len(preds)
    return smape_val


def keras_mape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def keras_SMAPE(y_true, y_pred):
    #Symmetric mean absolute percentage error
    return 200 * K.mean(K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true) + K.epsilon()), axis=-1)
