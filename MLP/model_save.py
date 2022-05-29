
def tf_save(model, filepath:str):
    """ This functions saves:
            - The architectire of the model, allowing to re-create the model
            - The weights of the model
            - The training configuration(loss, opt)
            - The state of the optimizer, allowiing to resume training exacly where you left off
        Arguments:
            model: Tensorflow model
            filepath: saved filepath (eg. 'models/medical_trial_model.h5')
    """
    model.save(filepath)
     
def model_to_json(model, filepath):
    """Saving the model as JSON:
            - If you only need the architecture of a model
            - Not its weights or training configuration

    Args:
        model (_type_): _description_
        filepath (str):
    """
    model_json = model.to_json()
    with open(filepath, 'w') as f:
        f.write(model_json)

def save_weights(model,filepath:str):
    model.save_weights(filepath)

#  Method `model.to_yaml()` has been removed due to security risk of arbitrary code execution. Please use `model.to_json()` instead.
"""def model_to_yaml(model):
    model_yaml =model.to_yaml()
    with open('models/medical_trial_yaml_model.txt', 'w') as f:
        f.write(model_yaml)"""