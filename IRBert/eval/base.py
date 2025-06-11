import abc

__all__ = ['BaseReport']

class BaseReport(abc.ABC):
     def __init__(self, prediction, ground_truth, metrics, 
             block_length = 15, terminal_length = 120, eps = 1e-8):

         self.metrics = metrics
         self.block_length = block_length
         self.terminal_length = terminal_length

         self.eps = eps

         self.results = self.evaluate(prediction, ground_truth)

     @property
     def metrics(self):
         return self._metrics

     @metrics.setter
     def metrics(self, metrics):
         assert isinstance(metrics, (list, tuple))
         self._metrics = metrics
         return None

     @property
     def results(self):
         return self._results

     @results.setter
     def results(self, results):
         assert isinstance(results, dict)
         for m in results:
             assert m in self.metrics

         self._results = results

         return None

     @property
     def block_length(self):
         return self._block_length

     @block_length.setter
     def block_length(self, block_length):
         assert isinstance(block_length, int)
         assert block_length > 0
         self._block_length = block_length
         return None

     @property
     def terminal_length(self):
         return self._terminal_length

     @terminal_length.setter
     def terminal_length(self, terminal_length):
         assert isinstance(terminal_length, int)
         assert terminal_length > 0
         assert terminal_length > self.block_length

         self._terminal_length = terminal_length
         return None

     @property
     def eps(self):
         return self._eps

     @eps.setter
     def eps(self, eps):
         assert isinstance(eps, float)
         assert eps > 0.
         self._eps = eps
         return None

     def __repr__(self):
         return '{0}(metics={1})'.format(self.__class__.__name__, 
                 self.metrics)

     def get_metric(self, metric):
         assert metric in self.metrics
         return self.results[metric]

     @abc.abstractmethod
     def evaluate(self, prediction, ground_truth):
         raise NotImplementedError()

     @abc.abstractmethod
     def summary(self):
         raise NotImplementedError()

     @abc.abstractmethod
     def save_as_csv(self):
         raise NotImplementedError()


