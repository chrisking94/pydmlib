import base

#print(object.__dict__['__str__'])
def kk(aa, *args ,**kwargs):
    print(locals())
    print(*args)
    print(kwargs)
print(hasattr(kk, 'aa'))
print(kk.__dict__)
kk(1,2,3,aa=5,bb=6)
#base.test()