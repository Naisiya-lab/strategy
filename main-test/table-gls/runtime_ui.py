"""Keep model/native/worker output in a log; progress uses a dedicated terminal FD."""
import os, sys, traceback
from contextlib import contextmanager

@contextmanager
def model_output(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush(); sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    terminal = os.fdopen(os.dup(2), "w", buffering=1)
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            try:
                yield terminal
            except BaseException:
                traceback.print_exc(file=log)
                print(f"Run failed or interrupted. See log: {log_path}", file=terminal, flush=True)
                raise
            finally:
                sys.stdout.flush(); sys.stderr.flush()
    finally:
        os.dup2(saved_out, 1); os.dup2(saved_err, 2)
        os.close(saved_out); os.close(saved_err); terminal.close()
