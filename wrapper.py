from concurrent.futures import ProcessPoolExecutor
from functools import partial

def _process_batch(batch, func):
    # This function is now defined at the module level and is pickleable.
    return [func(item) for item in batch]

class BatchingPool:
    def __init__(self, pool, batch_size=10):
        self.pool = pool
        self.batch_size = batch_size

    def map(self, func, iterable):
        # Convert the iterable to a list and group items into batches.
        items = list(iterable)
        batches = [items[i:i+self.batch_size] for i in range(0, len(items), self.batch_size)]
        # Use functools.partial to pass the likelihood function to _process_batch.
        partial_func = partial(_process_batch, func=func)
        results_batches = self.pool.map(partial_func, batches)
        # Flatten the list of results.
        results = [item for batch in results_batches for item in batch]
        return results

    def close(self):
        self.pool.shutdown()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()