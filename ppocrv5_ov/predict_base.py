import openvino as ov

class PredictBase(object):
    """
    A base class for prediction using OpenVINO.
    Methods
    -------
    __init__():
        Initializes the PredictBase object.
    get_compiled_model(model_dir, device):
        Compiles the model using OpenVINO for the specified device.
    Parameters
    ----------
    model_dir : str
        The directory where the model is located.
    device : str
        The device on which to compile the model (e.g., 'CPU', 'GPU', 'AUTO'...).
    """
    def __init__(self):
        pass

    def get_compiled_model(self, model_dir, device):
        """
        Compiles a model for a specified device using OpenVINO.
        Args:
            model_dir (str): The directory path where the model is located.
            device (str): The target device to compile the model for (e.g., 'CPU', 'GPU', 'AUTO'...).
        Returns:
            CompiledModel: The compiled OpenVINO model.
        """

        compile_model = ov.compile_model(model_dir, device)

        return compile_model