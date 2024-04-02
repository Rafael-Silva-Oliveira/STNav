def pass_analysis_pipeline(func):
    def wrapper(STNavCorePipeline, *args, **kwargs):
        return func(STNavCorePipeline, *args, **kwargs)

    return wrapper
