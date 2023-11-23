from . import logmanager

def getlogger(cls):
    name = cls
    if not isinstance(cls, str):
        if hasattr(cls, "__class__"):
            if hasattr(cls.__class__, "__name__"):
                name = cls.__class__.__name__
    return logmanager.get(name)
