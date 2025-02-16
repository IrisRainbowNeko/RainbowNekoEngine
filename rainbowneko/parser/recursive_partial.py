from functools import partial


class RecursivePartial(partial):
    __slots__ = "func", "args", "keywords", "add_sub"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(RecursivePartial, cls).__new__(cls, func)

        self.func = func
        self.args = [args]
        self.keywords = [keywords]
        self.add_sub = False
        return self

    def __call__(self, /, *args, **keywords):
        if self.add_sub:
            self.args.append(args)
            self.keywords.append(keywords)
            self.add_sub = False
            return self
        else:
            keywords = {**self.keywords[0], **keywords}
            res = self.func(*self.args[0], *args, **keywords)
            for args_i, kwargs_i in zip(self.args[1:], self.keywords[1:]):
                if len(args_i) == 2 and args_i[0] is getattr:
                    res = getattr(res, args_i[1])
                else:
                    res = res(*args_i, **kwargs_i)
            return res

    def __repr__(self):
        fdict = lambda d: ', '.join(f'{k}={v}' for k, v in d.items())
        flist = lambda l: ', '.join(f'{k}' for k in l) + ', ' if len(l) > 0 else ''

        pstr = f'{self.__class__.__name__}[{self.func}({flist(self.args[0])}{fdict(self.keywords[0])})'
        for args_i, kwargs_i in zip(self.args[1:], self.keywords[1:]):
            if len(args_i) == 2 and args_i[0] is getattr:
                pstr += f'.{args_i[1]}'
            else:
                pstr += f'({flist(args_i)}{fdict(kwargs_i)})'
        return pstr+']'
