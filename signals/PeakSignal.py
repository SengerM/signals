from .Signal import Signal
import numpy as np
from scipy import interpolate
import warnings
from scipy.stats import median_abs_deviation

warnings.filterwarnings('ignore', message='Mean of empty slice') # Don't know exactly where this warning is happening, but it does not affect the results.

def guess_peak_sign(samples):
	return 1 if samples[:11].mean()<np.mean((samples.max(),samples.min())) else -1

class PositivePeakSignal(Signal):
	"""Class intended to deal with 'single peak signals', i.e. a signal 
	that is 'zero zero PEAK zero zero' where 'zero' can be noise. It handles
	only positive peaks."""
	
	@property
	def t_max(self) -> float:
		"""Returns the time at which the maximum sample is found, i.e.
		the peak time."""
		if not hasattr(self, '_t_max'):
			self._t_max = self._time[np.argmax(self._samples)]
		return self._t_max
	
	@property
	def peak_start_index(self) -> int:
		"""Returns the index of the sample where the peak starts."""
		if not hasattr(self, '_peak_start_index'):
			try:
				peak_index = np.argmax(self._samples)
				median_before_peak = np.nanmedian(self._samples[:peak_index])
				std_before_peak = median_abs_deviation(self._samples[:peak_index])*1.4826 # https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
				indices_where_signal_is_lower_than_median = np.squeeze(np.where(self._samples<=median_before_peak+std_before_peak))
				self._peak_start_index = indices_where_signal_is_lower_than_median[np.squeeze(np.where(indices_where_signal_is_lower_than_median<peak_index))[-1]]
			except Exception:
				self._peak_start_index = None
		return self._peak_start_index
	
	@property
	def peak_start_time(self) -> float:
		"""Returns the time at which the peak starts. The current 
		implementation returns the time of the sample with 
		`self.peak_start_index`.
		"""
		if self.peak_start_index is not None:
			return self.time[self.peak_start_index]
		else:
			return float('NaN')
	
	@property
	def _baseline_positive_peak(self) -> float:
		"""Returns the baseline of the signal, i.e. the value at which 
		it was stable before the peak started.
		"""
		if not hasattr(self, '_baseline'):
			try:
				self._baseline = np.nanmean(self._samples[:self.peak_start_index-1])
			except Exception:
				self._baseline = float('NaN')
		return self._baseline
	
	@property
	def _amplitude_positive_peak(self) -> float:
		"""Returns the amplitude of the signal defined as the difference 
		between the maximum value and the baseline.
		"""
		if not hasattr(self, '_amplitude'):
			try:
				self._amplitude = (self._samples - self._baseline_positive_peak).max()
			except Exception:
				self._amplitude = float('NaN')
		return self._amplitude
	
	@property
	def noise(self) -> float:
		"""Returns the noise of the signal defined as the standard 
		deviation of the samples before the peak starts, or `float('NaN')` 
		if it cannot be determined.
		"""
		if not hasattr(self, '_noise'):
			try:
				with warnings.catch_warnings():
					warnings.filterwarnings('ignore')
					self._noise = np.nanstd(self._samples[:self.peak_start_index-1])
			except Exception:
				self._noise = float('NaN')
		return self._noise
	
	@property
	def SNR(self) -> float:
		"""Returns the signal to noise ratio defined as amplitude/noise."""
		if not hasattr(self, '_SNR'):
			if self.noise != 0:
				self._SNR = self._amplitude_positive_peak/self.noise
			else:
				self._SNR = float('NaN')
		return self._SNR
	
	@property
	def rise_time(self) -> float:
		"""Returns the rise time defined as the time spent by the signal 
		to go from 10 % to 90 %.
		"""
		if not hasattr(self, '_rise_time'):
			try:
				self._rise_time = self.find_time_at_rising_edge(90) - self.find_time_at_rising_edge(10)
			except Exception:
				self._rise_time = float('NaN')
		return self._rise_time
	
	@property
	def rising_edge_indices(self) -> list:
		"""Returns a list of integer numbers corresponding to the indices 
		of the `time` and `samples` arrays where the rising edge is located. 
		The rising edge is considered to start at 10 % and end at 90 %. 
		If the rising edge cannot be found, returns an empty list.
		"""
		if not hasattr(self, '_rising_edge_indices'):
			try:
				self._rising_edge_indices = self.find_rising_edge_indices(low=10,high=90)
			except Exception:
				self._rising_edge_indices = []
		return self._rising_edge_indices
	
	@property
	def falling_edge_indices(self) -> list:
		"""Returns a list of integer numbers corresponding to the indices 
		of the `time` and `samples` arrays where the falling edge is located. 
		The falling edge is considered to start at 10 % and end at 90 %. 
		If the falling edge cannot be found, returns an empty list.
		"""
		if not hasattr(self, '_falling_edge_indices'):
			try:
				self._falling_edge_indices = self.find_falling_edge_indices(low=10,high=90)
			except Exception:
				self._falling_edge_indices = []
		return self._falling_edge_indices
	
	@property
	def time_over_noise(self) -> float:
		"""Returns the time the pulse spends over the noise value."""
		if not hasattr(self, '_time_over_noise'):
			try:
				self._time_over_noise = self.find_time_over_threshold(threshold = self.noise/self._amplitude_positive_peak*100)
			except Exception:
				self._time_over_noise = float('NaN')
		return self._time_over_noise
	
	@property
	def _peak_integral_positive_peak(self) -> float:
		"""Returns the integral under the peak. The peak start is defined 
		as that point where the signal goes outside of the noise band, 
		and the end is the moment in which it goes back inside the noise 
		band.
		"""
		if not hasattr(self, '_peak_integral'):
			try:
				peak_points = (self.time>=self.find_time_at_rising_edge(self.noise/self._amplitude_positive_peak*100))&(self.time<=self.find_time_at_falling_edge(self.noise/self._amplitude_positive_peak*100))
				self._peak_integral = np.trapz(x=self.time[peak_points], y=self._samples[peak_points]-self._baseline_positive_peak)
			except Exception:
				self._peak_integral = float('NaN')
		return self._peak_integral
	
	@property
	def _integral_from_baseline_positive_peak(self) -> float:
		"""Returns the integral of the whole signal taking as 0 the baseline. 
		This means that values above the baseline contribute positively
		and values below contribute negatively.
		"""
		if not hasattr(self, '_integral_from_baseline'):
			try:
				self._integral_from_baseline = np.trapz(x=self.time, y=self._samples-self._baseline_positive_peak)
			except Exception:
				self._integral_from_baseline = float('NaN')
		return self._integral_from_baseline
	
	def find_rising_edge_indices(self, low: float, high: float) -> list:
		"""Finds the rising edge of the signal. Returns a list of integers 
		corresponding to the indices of the rising edge between `low` % and `high` %.
		
		Parameters
		----------
		low: float
			Percentage to consider where the rising edge starts, e.g. 
			10 %.
		high: float
			Percentage to consider where the rising edge ends, e.g. 90 %.
		"""
		for name,x in {'low': low, 'high': high}.items():
			if not isinstance(x, (int, float)):
				raise TypeError(f'`{name}` must be a float number, but received object of type {type(x)}.')
		if not low < high:
			raise ValueError(f'`low` must be less than `high`, received low={low} and high={high}.')
		k = self._samples.argmax()
		k_start_rise = None
		k_stop_rise = None
		while k > 0:
			if self._samples[k] - self._baseline_positive_peak > self._amplitude_positive_peak*high/100:
				k_stop_rise = k+1
			if self._samples[k] - self._baseline_positive_peak < self._amplitude_positive_peak*low/100:
				k_start_rise = k
				break
			k -= 1
		if k_start_rise is None or k_stop_rise is None or k_start_rise == k_stop_rise:
			raise RuntimeError(f'Cannot find the rising edge of this signal.')
		return [k for k in range(k_start_rise, k_stop_rise)]
	
	def find_falling_edge_indices(self, low: float, high: float) -> list:
		"""Finds the falling edge of the signal. Returns a list of integers 
		corresponding to the indices of the falling edge between `low` % 
		and `high` %.
		
		Parameters
		----------
		low: float
			Percentage to consider where the falling edge starts, e.g. 
			10 %.
		high: float
			Percentage to consider where the falling edge ends, e.g. 90 %.
		"""
		for name,x in {'low': low, 'high': high}.items():
			if not isinstance(x, (int, float)):
				raise TypeError(f'`{name}` must be a float number, but received object of type {type(x)}.')
		if not low < high:
			raise ValueError(f'`low` must be less than `high`, received low={low} and high={high}.')
		k = self._samples.argmax()
		k_start_fall = None
		k_stop_fall = None
		while k < len(self._samples):
			if self._samples[k] - self._baseline_positive_peak > self._amplitude_positive_peak*high/100:
				k_start_fall = k
			if self._samples[k] - self._baseline_positive_peak < self._amplitude_positive_peak*low/100:
				k_stop_fall = k + 1
				break
			k += 1
		if k_start_fall is None or k_stop_fall is None:
			raise RuntimeError(f'Cannot find the falling edge of this signal.')
		return [k for k in range(k_start_fall, k_stop_fall)]
	
	def find_time_at_rising_edge(self, threshold: float) -> float:
		"""Given some threshold value (as a percentage) returns the time 
		at which the signal crosses such threshold within the rising edge. 
		The signal is linearly interpolated between samples.
		"""
		if not isinstance(threshold, (float, int)):
			raise TypeError(f'`threshold` must be a float number, received object of type {type(threshold)}.')
		if not 0 < threshold < 100:
			raise ValueError(f'`threshold` must be between 0 and 100, received {threshold}.')
		if np.isnan(self._amplitude_positive_peak):
			raise RuntimeError('Cannot find the amplitude of the signal.')
		if np.isnan(self._baseline_positive_peak):
			raise RuntimeError('Cannot find the baseline of the signal.')
		rising_edge_indices = self.find_rising_edge_indices(low=threshold, high=99)
		return float(interpolate.interp1d(
			x = self._samples[rising_edge_indices],
			y = self.time[rising_edge_indices],
		)(self._amplitude_positive_peak*threshold/100 + self._baseline_positive_peak))
	
	def find_time_at_falling_edge(self, threshold: float) -> float:
		"""Given some threshold value (as a percentage) returns the time 
		at which the signal crosses such threshold within the falling edge. 
		The signal is linearly interpolated between samples.
		"""
		if not isinstance(threshold, (float, int)):
			raise TypeError(f'`threshold` must be a float number, received object of type {type(threshold)}.')
		if not 0 < threshold < 100:
			raise ValueError(f'`threshold` must be between 0 and 100, received {threshold}.')
		if np.isnan(self._amplitude_positive_peak):
			raise RuntimeError('Cannot find the amplitude of the signal.')
		if np.isnan(self._baseline_positive_peak):
			raise RuntimeError('Cannot find the baseline of the signal.')
		falling_edge_indices = self.find_falling_edge_indices(low=threshold, high=99)
		return float(interpolate.interp1d(
			x = self._samples[falling_edge_indices],
			y = self.time[falling_edge_indices],
		)(self._amplitude_positive_peak*threshold/100 + self._baseline_positive_peak))
	
	def find_time_over_threshold(self, threshold: float) -> float:
		"""Returns the time over some threshold where `threshold` is a 
		percentage.
		"""
		if not isinstance(threshold, (float, int)):
			raise TypeError(f'`threshold` must be a number, received object of type {type(threshold)}.')
		if not 0 < threshold < 100:
			raise ValueError(f'`threshold` must be within 0 and 100, received {threshold}.')
		return self.find_time_at_falling_edge(threshold) - self.find_time_at_rising_edge(threshold)
	
	def get_peak_integral_fraction_time(self, fraction) -> float:
		""" Get fraction of signal collected after a certain time
		Added by A. Ilg and S. Burkhalter
		"""
		peak_points = (self.time >= self.find_time_at_rising_edge(self.noise/self.amplitude*100)) & (self.time <= self.find_time_at_falling_edge(self.noise/self.amplitude*100))
		times = self.time[peak_points]
		integrals = np.cumsum((self.samples[peak_points][:-1] - self.baseline) * np.diff(times))
		full_integral = integrals[-1]
		target_integral = fraction * full_integral
		
		# Find start and end times for target integral
		start_index = np.argmin(np.abs(integrals - target_integral))
		fraction_time = self.time[start_index]
		return fraction_time

class PeakSignal(PositivePeakSignal):
	"""Class to handle peaked signals, both positive and negative."""
	def __init__(self, time, samples, peak_polarity:str='positive'):
		"""Create a `PeakSignal` object.
		
		Arguments
		---------
		time: array like
			An array like object with the times for each sample.
		samples: array like
			An array like object with the samples.
		peak_polarity: str, default `'positive'`
			Options are `'positive'`, `'negative'` and `'guess'`. This
			defines how the peak will be searched.
		"""
		PEAK_POLARITY_OPTIONS = {'positive','negative','guess'}
		if peak_polarity not in PEAK_POLARITY_OPTIONS:
			raise ValueError(f'`peak_polarity` must be one of {PEAK_POLARITY_OPTIONS}, received {repr(peak_polarity)}. ')
		if peak_polarity == 'positive':
			self._peak_sign = 1
		elif peak_polarity == 'negative':
			self._peak_sign = -1
		elif peak_polarity == 'guess':
			self._peak_sign = guess_peak_sign(samples)
		
		super().__init__(
			time = time,
			samples = samples*self._peak_sign,
		)
	
	@property
	def samples(self):
		return self._samples*self._peak_sign
	
	@property
	def baseline(self) -> float:
		return self._baseline_positive_peak * self._peak_sign
	
	@property
	def amplitude(self) -> float:
		return self._amplitude_positive_peak * self._peak_sign
	
	@property
	def peak_integral(self) -> float:
		return self._peak_integral_positive_peak * self._peak_sign
	
	@property
	def integral_from_baseline(self) -> float:
		return self._integral_from_baseline_positive_peak * self._peak_sign

def draw_in_plotly(signal, fig=None, baseline=True, noise=True, amplitude=True, rise_time=True, time_over_noise=True, peak_integral=True, peak_start_time=True):
	"""Plot the signal along with the different quantities. `fig` is a 
	plotly figure.
	"""
	import plotly.graph_objects as go
	if not isinstance(signal, PeakSignal):
		raise TypeError(f'`signal` must be an instance of {repr(PeakSignal)}, received object of type {repr(type(signal))}.')
	if fig is None:
		fig = go.Figure()
	if type(fig) != type(go.Figure()):
		raise TypeError(f'`fig` must be a plotly figure, received object of type {repr(type(fig))}.')
	
	fig.add_trace(
		go.Scatter(
			x = signal.time,
			y = signal.samples,
			mode = 'lines+markers',
			name = 'Signal',
		)
	)
	if peak_integral == True:
		try:
			t_start = signal.find_time_at_rising_edge(signal.noise/abs(signal.amplitude)*100)
			t_stop = signal.find_time_at_falling_edge(signal.noise/abs(signal.amplitude)*100)
		except:
			t_start = float('NaN')
			t_stop = float('NaN')
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fig.add_trace(
				go.Scatter(
					x = [t_start] + list(signal.time[(signal.time>t_start)&(signal.time<t_stop)]) + [t_start + signal.time_over_noise] + [t_stop,t_start] + [t_start],
					y = [signal(t_start)] + list(signal.samples[(signal.time>t_start)&(signal.time<t_stop)]) + [signal(t_start + signal.time_over_noise)] + 2*[signal.baseline] + [signal(t_start)],
					name = f'Integral ({signal.peak_integral:.2e})',
					fill = 'toself',
					mode = 'none',
					line = dict(color='#ff6363'),
				)
			)
	if baseline == True:
		fig.add_trace(
			go.Scatter(
				x = [min(signal.time), max(signal.time)],
				y = [signal.baseline]*2,
				mode = 'lines',
				name = f'Baseline ({signal.baseline:.2e})',
				line = dict(color='black', dash='dash'),
			)
		)
	if noise == True:
		fig.add_trace(
			go.Scatter(
				x = [min(signal.time), max(signal.time)] + [float('NaN')] + [max(signal.time), min(signal.time)],
				y = [signal.baseline + signal.noise, signal.baseline + signal.noise] + [float('NaN')] + [signal.baseline - signal.noise, signal.baseline - signal.noise],
				mode = 'lines',
				name = f'Noise ({signal.noise:.2e})',
				line = dict(color='black', width=.7, dash='dash'),
			)
		)
	if amplitude == True:
		fig.add_trace(
			go.Scatter(
				x = [signal.t_max]*2,
				y = [signal.baseline, signal.baseline + signal.amplitude],
				name = f'Amplitude ({signal.amplitude:.2e})',
				mode = 'lines+markers',
				line = dict(color='rgba(50, 163, 39, .7)'),
				marker = dict(size=11),
			)
		)
	if rise_time == True:
		try:
			t_start_rise = signal.find_time_at_rising_edge(threshold=10)
		except:
			t_start_rise = float('NaN')
		fig.add_trace(
			go.Scatter(
				x = [t_start_rise, t_start_rise+signal.rise_time, t_start_rise+signal.rise_time, t_start_rise, t_start_rise],
				y = signal.baseline + np.array([signal.amplitude*.1, signal.amplitude*.1, signal.amplitude*.9, signal.amplitude*.9, signal.amplitude*.1]),
				name = f'Rise time ({signal.rise_time:.2e})',
				mode = 'lines',
				line = dict(color='rgba(196, 0, 173, .5)'),
			)
		)
	if time_over_noise == True:
		threshold = signal.noise/abs(signal.amplitude)*100
		try:
			t_start = signal.find_time_at_rising_edge(threshold)
		except:
			t_start = float('NaN')
		fig.add_trace(
			go.Scatter(
				x = [t_start,t_start + signal.time_over_noise],
				y = 2*[signal.baseline+threshold/100*signal.amplitude],
				name = f'Time over noise ({signal.time_over_noise:.2e})',
				mode = 'lines+markers',
				line = dict(color='#bf6c00', dash='dashdot'),
				marker = dict(size=11),
			)
		)
	if peak_start_time == True and not np.isnan(signal.peak_start_time):
		fig.add_vline(
			x = signal.peak_start_time,
			line_color = 'black',
			line_dash = 'dashdot',
			line_width = .5,
			annotation_text = f'Peak start time = {signal.peak_start_time:.2e}',
			annotation_textangle = -90,
			annotation_position = 'top left',
		)

	return fig

def compress_PeakSignal_V230507(signal:PeakSignal)->dict:
	"""This is a (almost) lossless compression method that simply drops all the
	samples that are in front of the peak, previously calculating the 
	baseline and noise amplitude. From the peak start signal on to the end,
	all the samples as well as the time is kept. This allows to reconstruct
	the waveform afterward "without loss", meaning that all the information
	in the peak is original, as well as whatever happened after the peak.
	Before the peak, where we assume there was no causal effect from the
	peak and only noise, new fake samples can be generated easily with
	the same baseline and noise amplitude as white noise.
	To recover the `PeakSignal` object (i.e. to decompress this) use
	the function `decompress_signal_V230507`.
	"""
	if not isinstance(signal, PeakSignal):
		raise TypeError(f'`signal` must be an instance of {repr(PeakSignal)}, received object of type {type(signal)}. ')
	try:
		return dict(
			compressed_time = tuple(signal.time[signal.peak_start_index:]),
			compressed_samples = tuple(signal.samples[signal.peak_start_index:]),
			noise = signal.noise,
			baseline = signal.baseline,
			number_of_samples_removed = signal.peak_start_index-1,
			compression_algorithm = 'compress_signal_V230507',
		)
	except Exception as e: # This happens when it cannot be compressed for some reason.
		return dict(
			compressed_time = tuple(signal.time),
			compressed_samples = tuple(signal.samples),
			noise = float('NaN'),
			baseline = float('NaN'),
			number_of_samples_removed = 0,
			compression_algorithm = 'compress_signal_V230507',
		)
	
def decompress_PeakSignal_V230507(compressed_signal:dict)->PeakSignal:
	"""Recover a signal that was compressed with `compress_PeakSignal_V230507`."""
	if not isinstance(compressed_signal, dict):
		raise TypeError(f'`compressed_signal` must be a dictionary, received object of type {type(compressed_signal)}. ')
	if compressed_signal.get('compression_algorithm') != 'compress_signal_V230507':
		raise ValueError(f'Cannot decompress a compressed signal whose `compression_algorithm` field is {repr(compressed_signal.get("compression_algorithm"))}. ')
	
	n_samples_to_add = compressed_signal['number_of_samples_removed']
	if n_samples_to_add == 0: # This means that there was no compression at all.
		return PeakSignal(
			time = compressed_signal['compressed_time'],
			samples = compressed_signal['compressed_samples'],
		)
	sampling_period = np.diff(compressed_signal['compressed_time']).mean()
	add_time = np.linspace(compressed_signal['compressed_time'][0]-sampling_period*(n_samples_to_add+1), compressed_signal['compressed_time'][0]-sampling_period, n_samples_to_add)
	add_samples = np.random.randn(n_samples_to_add)
	add_samples *= compressed_signal['noise']/add_samples.std()
	add_samples += compressed_signal['baseline'] - add_samples.mean()
	return PeakSignal(
		time = np.concatenate([add_time, compressed_signal['compressed_time']]),
		samples = np.concatenate([add_samples, compressed_signal['compressed_samples']]),
	)
