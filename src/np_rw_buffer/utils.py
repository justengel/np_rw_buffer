import functools


__all__ = ['make_thread_safe']


def make_thread_safe(lock_varname="lock", func=None):
    """Decorate a function making it threadsafe by using the threading lock that matches the lock_varname.

    Args:
        lock_varname (str/method)['lock']: Threading lock variable name or
            a function to decorate with 'lock' variable being a threading.Lock
        func (function/method) [None]: Function to wrap.

    Returns:
        wrap (function): Function that was decorated/wrapped or a function that will decorate a function.
    """
    if not isinstance(lock_varname, str):
        # Function was given decorate the function
        func = lock_varname
        lock_varname = 'lock'

    if func is None:
        # Return a decorator
        def real_decorator(func):
            return make_thread_safe(lock_varname, func)
        return real_decorator

    elif isinstance(func, property):
        fget = None
        fset = None
        fdel = None
        if func.fget:
            fget = make_thread_safe(lock_varname, func.fget)
        if func.fset:
            fset = make_thread_safe(lock_varname, func.fset)
        if func.fdel:
            fdel = make_thread_safe(lock_varname, func.fdel)
        return property(fget, fset, fdel, func.__doc__)

    else:
        # Return the new function which wraps the old function
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with getattr(args[0], lock_varname):
                return func(*args, **kwargs)
        return wrapper
