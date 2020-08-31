import time
from functools import wraps
from rich.progress import track

def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print('function:%r took: %2.4f sec' % (f.__name__,  end - start))
        return result
    return wrapper


@timing
def test_timeing():
    for i in track(range(1000000)):
        if i % 100000 == 0:
            print(i)

if __name__ == '__main__':
    test_timeing()