def class_to_dict(cls):
    def props(cls):
        return [[i, getattr(cls, i)] for i in cls.__dict__.keys() if i[:1] != '_']

    return dict(props(cls))



def get_model_device(model):
    return next(model.parameters()).device