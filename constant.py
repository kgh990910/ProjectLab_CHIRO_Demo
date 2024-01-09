class _constant:
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise Exception("할당불가")
        self.__dict__[name] = value
        
    def __delattr__(self, name):
        if name in self.__dict__:
            raise Exception("삭제 불가")

import sys
sys.modules[__name__] = _constant