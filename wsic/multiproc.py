import multiprocessing.queues
from typing import Any


class Queue(multiprocessing.queues.Queue):
    """A multiprocessing.Queue subclass with a shared counter.

    The `multiprocessing.Queue.qsize` is unreliable and may raise
    `NotImplementedError` on Unix platforms like macOS where
    `sem_getvalue()` is not implemented. Here we use a shared counter
    instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the queue."""
        super().__init__(*args, ctx=multiprocessing.get_context(), **kwargs)
        self.counter = multiprocessing.Value("i", 0)

    def put(self, *args, **kwargs) -> None:
        """Put an item into the queue."""
        with self.counter.get_lock():
            self.counter.value += 1
        return super().put(*args, **kwargs)

    def get(self, *args, **kwargs) -> Any:
        """Remove and return an item from the queue."""
        value = super().get(*args, **kwargs)
        with self.counter.get_lock():
            self.counter.value -= 1
        return value  # noqa: R504

    def qsize(self) -> int:
        """Return the number of items in the queue."""
        return self.counter.value

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return self.qsize() == 0

    def clear(self) -> None:
        """Clear the queue."""
        while not self.empty():
            self.get()

    def __getstate__(self) -> Any:
        """Return the state of the queue."""
        return (self.counter, super().__getstate__())

    def __setstate__(self, state: Any) -> None:
        """Restore the state of the queue."""
        self.counter, state = state
        super().__setstate__(state)
        self._after_fork()
