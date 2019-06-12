from functools import wraps
import  time
import  pickle
from config import log



def func_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        log.info('Total time running for function %s: %s seconds.' % (func.__doc__, t2 - t1))
        return result
    return wrapper

class tst(object):
    pass

@func_timer
def test():
    pass

if __name__ == '__main__':
    # with open('svm/saved_models/tst','wb') as f:
    #     pickle.dump(func_timer, f)
    # with open('svm/saved_models/tst','rb') as f:
    #     func = pickle.load(f)
    #     print(func.__doc__)
    #     print(func.__name__)
    test()