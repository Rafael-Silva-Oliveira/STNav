def pass_STNavCore_params(func):
    def wrapper(STNavCorePipeline, *args, **kwargs):
        return func(STNavCorePipeline, *args, **kwargs)

    return wrapper
