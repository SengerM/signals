import numpy as np
from scipy import interpolate, integrate

class Signal:
	"""Most basic class of a signal."""
	def __init__(self, time, samples):
		"""
		- time: array of times.
		- samples: array with samples of the signal.
		"""
		if len(time) != len(samples):
			raise ValueError(f'`len(time) == len(samples)` must be true, but received `len(time)=={len(time)}` and `len(samples)=={len(samples)}.')
		self._time = np.array(time)
		self._samples = np.array(samples)
	
	@property
	def time(self):
		"""Returns the array of times."""
		return self._time
	
	@property
	def samples(self):
		"""Returns the array of samples."""
		return self._samples
	
	def __call__(self, time: float, interpolation='linear'):
		"""Returns the value of the signal at any time."""
		if interpolation == 'linear':
			return interpolate.interp1d(self.time, self.samples)(time)
		elif isinstance(interpolation, int):
			return interpolate.interp1d(self.time, self.samples, kind=interpolation)(time)
		else:
			raise NotImplementedError(f'Interpolation {repr(interpolation)} not implemented.')
