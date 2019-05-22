def class_to_dict(cls):
    def props(cls):
        return [[i, getattr(cls, i)] for i in cls.__dict__.keys() if i[:1] != '_']

    return dict(props(cls))


def calculate_original_img_size(origin_size: int, upscale_factor: int) -> int:
    """
    元の画像サイズを縮小拡大したいときに元の画像をどの大きさに
    resize する必要があるかを返す関数

    例えば 202 px の画像を 1/3 に縮小することは出来ない(i.e. 3の倍数ではない)ので
    事前に 201 px に縮小しておく必要がありこの関数はその計算を行う
    すなわち

    calculate_original_img_size(202, 3) -> 201

    となる

    Args:
        origin_size:
        upscale_factor:

    Returns:

    """
    return origin_size - (origin_size % upscale_factor)


def get_model_device(model):
    return next(model.parameters()).device
