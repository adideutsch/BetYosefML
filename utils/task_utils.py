import functools, pickle, os.path, time, hashlib

CACHE_FILENAME = "cache/parsing_cache_%s.pickle"


class TimeCounter():
    def __init__(self, name):
        self.initial_time = time.time()
        self.name = name
    def start(self):
        self.initial_time = time.time()
    def stop(self):
        self.interval = time.time() - self.initial_time
    def stop_and_report(self):
        self.stop()
        print("~%d seconds" % (int(self.interval)))

def load_pickle(call_id):
    # If file doesn't exist create initial one
    filename = CACHE_FILENAME % (call_id)
    if not os.path.isfile(filename):
        with open(filename, 'wb') as fobj:
            pickle.dump({}, fobj, pickle.HIGHEST_PROTOCOL)

    with open(filename, 'rb') as fobj:
        data = pickle.load(fobj)
    return data

def dump_pickle(data, call_id):
    filename = CACHE_FILENAME % (call_id)
    with open(filename, 'wb') as fobj:
        pickle.dump(data, fobj, pickle.HIGHEST_PROTOCOL)

def load_from_cache(call_id):
    cache = load_pickle(call_id)
    if call_id in cache:
        return cache[call_id]
    else:
        return False

def dump_to_cache(call_id, result):
    cache = load_pickle(call_id)
    cache[call_id] = result
    dump_pickle(cache, call_id)

def timed_task(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        stopwatch = TimeCounter(f.__name__)
        stopwatch.start()
        # print("Task <%s>:" % (f.__name__))
        result = f(*args, **kwds)
        print("Task <%s> took " % (f.__name__), end="")
        stopwatch.stop_and_report()
        return result
    return wrapper

def cached_task(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        call_id = hashlib.md5(str((f.__name__, str(args), str(kwds))).encode('utf-8')).hexdigest()
        cache_data = load_from_cache(call_id)
        if cache_data != False:
            print("Task <%s> resolved from cache!" % (f.__name__))
            return cache_data
        result = f(*args, **kwds)
        dump_to_cache(call_id, result)
        return result
    return wrapper
