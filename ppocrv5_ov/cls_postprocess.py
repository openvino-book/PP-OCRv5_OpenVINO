import numpy as np

class ClsPostProcess(object):
    """
    ClsPostProcess is a class for post-processing classification results.
    Attributes:
        label_list (list or dict): A list or dictionary of labels. If None, 
                                   labels will be generated as {idx: idx}.
        key (str): A key to extract specific predictions from the input.
    Methods:
        __init__(label_list=None, key=None, **kwargs):
            Initializes the ClsPostProcess object with optional label_list and key.
        __call__(preds, label=None, *args, **kwargs):
            Processes the predictions and returns decoded output.
            Args:
                preds (numpy.ndarray): The predictions from the model.
                label (list, optional): The ground truth labels. Defaults to None.
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.
            Returns:
                decode_out (list): A list of tuples containing the label and its corresponding prediction score.
                If label is provided, returns a tuple (decode_out, label) where label is a list of tuples containing 
                the ground truth label and a score of 1.0.
    """

    def __init__(self, label_list=None, key=None):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list
        self.key = key

    def __call__(self, preds, label=None):
        if preds.ndim != 2:
            if self.key in preds:
                preds = preds[self.key]
            else:
                raise KeyError(f"Key '{self.key}' not found in predictions.")
        label_list = self.label_list or {idx: idx for idx in range(preds.shape[-1])}
        label_list = self.label_list
        if label_list is None:
            label_list = {idx: idx for idx in range(preds.shape[-1])}

        pred_idxs = preds.argmax(axis=1)
        decode_out = [(label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        if label is not None:
            if not isinstance(label, list) or not all(isinstance(idx, int) for idx in label):
                raise ValueError("The 'label' parameter must be a list of indices.")
            label = [(label_list[idx], 1.0) for idx in label]
            return decode_out, label
