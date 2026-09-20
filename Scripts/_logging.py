"""
Console-log capture - tees everything printed to the terminal into Logs/.

A full training run prints its per-epoch numbers, the classification report and the
confusion matrix to the terminal and nowhere else; results_<run>.json keeps the final
metrics but not the training curve. Closing the terminal therefore loses the only record
of how validation accuracy moved from epoch to epoch, which is exactly what you need to
see whether a run over-fitted and at which epoch the saved checkpoint was taken.

start("maf_full") duplicates stdout and stderr into Logs/maf_full_<timestamp>.log and
returns the path. The terminal still behaves exactly as before - the tee writes through to
the real stream first, and delegates isatty(), so tqdm still draws a live progress bar.

Progress bars are collapsed on the way to the file. tqdm redraws a bar by returning to the
start of the line with a carriage return, so writing the raw stream out would put several
hundred near-identical lines in the log for every epoch. Only the text after the last '\r'
of each line is kept, which is that bar's final state - one line per bar, the same thing
you would see on screen once it finished.

Usage - one line in an entry point, after the arguments are parsed:

    import _logging
    _logging.start(args.run_name)
"""
import atexit
import datetime
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOG_DIR = os.path.join(ROOT_DIR, "Logs")

_started = None


class _Tee:
    """Write-through proxy: the real stream first, then the log file."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle
        self._pending = ""

    def write(self, text):
        written = self._stream.write(text)
        if self._handle is None:  # detached at exit; terminal only from here on
            return written
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            # Keep only what survives the last carriage return - a finished progress bar.
            self._handle.write(line.rsplit("\r", 1)[-1] + "\n")
        self._handle.flush()
        return written

    def flush(self):
        self._stream.flush()
        if self._handle is not None:
            self._handle.flush()

    def detach(self):
        """Flush any trailing partial line, then stop writing to the file.

        tqdm closes its progress bars from __del__ during interpreter shutdown, which can
        land after the atexit hook has finished with the log. Detaching rather than
        leaving a closed handle in place means those late writes still reach the terminal
        and raise nothing.
        """
        if self._handle is None:
            return
        if self._pending:
            self._handle.write(self._pending.rsplit("\r", 1)[-1] + "\n")
            self._pending = ""
        self._handle.flush()
        self._handle = None

    # tqdm asks the stream these; answer for the terminal underneath, not the file, so the
    # progress bar keeps drawing the way it does without logging.
    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


def start(run_name=None, log_dir=None):
    """Begin teeing stdout/stderr to a timestamped file. Returns its path.

    Calling this twice in one process is a no-op - the first log wins.
    """
    global _started
    if _started is not None:
        return _started

    directory = log_dir or LOG_DIR
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "{}_{}.log".format(run_name or "run", stamp)
    path = os.path.join(directory, name)

    handle = open(path, "w", encoding="utf-8", buffering=1)
    handle.write("# started  : {}\n".format(datetime.datetime.now().isoformat(timespec="seconds")))
    handle.write("# command  : {}\n".format(" ".join(sys.argv)))
    handle.write("# cwd      : {}\n".format(os.getcwd()))
    handle.write("# python   : {}\n".format(sys.version.split()[0]))
    handle.write("#\n")
    handle.flush()

    out, err = _Tee(sys.stdout, handle), _Tee(sys.stderr, handle)
    sys.stdout, sys.stderr = out, err

    def _finish():
        out.detach()
        err.detach()
        handle.write("#\n# finished : {}\n".format(
            datetime.datetime.now().isoformat(timespec="seconds")))
        handle.flush()
        handle.close()

    atexit.register(_finish)
    _started = path
    print("Console log     ->", path)
    return path
