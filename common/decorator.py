from flask import request
import copy

class params:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
       
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print('preprocessing')
            print('preprocessing configuration', self.args, self.kwargs)
            reduce_kwargs = copy.copy(self.kwargs)
            for item in self.kwargs.items():
                value = request.args.get(item[0], item[1])
                reduce_kwargs[item[0]]= value
            reduce_args = self.args
            if len(self.args):
                reduce_args = tuple(reduce_kwargs.values())
            return func(*reduce_args, **reduce_kwargs)
        return wrapper