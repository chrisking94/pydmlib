import base

#print(object.__dict__['__str__'])

def f1(self, b=2):
    print(b)
    return 'OK!'

def wrap(func):
    def fwrapped(*args, **kwargs):
        print(func.__code__.co_varnames)
        print(args,kwargs)
        return func(*args, **kwargs)
    return fwrapped

f = wrap(f1)
print(object.__bases__)

class B():
    pass

setattr(B,'f1',f1)
b = B()
print(b.f1(3))
base.test()