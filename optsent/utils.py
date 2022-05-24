import cProfile
import pathlib

# needs testing coverage


def profile(func):
    fname = pathlib.Path(__file__).parents[1] / "profile" / f"func={func.__name__}.prof"
    fname.parent.mkdir(parents=True, exist_ok=True)

    def wrapped(*args, **kwargs):
        prof = cProfile.Profile()
        prof.enable()
        val = prof.runcall(func, *args, **kwargs)
        prof.disable()
        prof.dump_stats(fname)
        return val

    return wrapped
