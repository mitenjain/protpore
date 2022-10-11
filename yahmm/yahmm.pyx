#!/usr/bin/env python2.7
# yahmm.pyx: Yet Another Hidden Markov Model library
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )
#          Adam Novak ( anovak1@ucsc.edu )

"""
For detailed documentation and examples, see the README.
"""

cimport cython
from cython.view cimport array as cvarray
from libc.math cimport log as clog, sqrt as csqrt, exp as cexp
import math, random, itertools as it, sys, bisect
import networkx
import scipy.stats, scipy.sparse, scipy.special

if sys.version_info[0] > 2:
	# Set up for Python 3
	from functools import reduce
	xrange = range
	izip = zip
else:
	izip = it.izip

import numpy
cimport numpy

from matplotlib import pyplot

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

# Useful speed optimized functions
cdef inline double _log ( double x ):
	'''
	A wrapper for the c log function, by returning negative input if the
	input is 0.
	'''

	return clog( x ) if x > 0 else NEGINF

cdef inline int pair_int_max( int x, int y ):
	'''
	Calculate the maximum of a pair of two integers. This is
	significantly faster than the Python function max().
	'''

	return x if x > y else y

cdef inline double pair_lse( double x, double y ):
	'''
	Perform log-sum-exp on a pair of numbers in log space..  This is calculated
	as z = log( e**x + e**y ). However, this causes underflow sometimes
	when x or y are too negative. A simplification of this is thus
	z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
	the inputs are infinity, return infinity, and if either of the inputs
	are negative infinity, then simply return the other input.
	'''

	if x == INF or y == INF:
		return INF
	if x == NEGINF:
		return y
	if y == NEGINF:
		return x
	if x > y:
		return x + clog( cexp( y-x ) + 1 )
	return y + clog( cexp( x-y ) + 1 )

# Useful python-based array-intended operations
def log(value):
	"""
	Return the natural log of the given value, or - infinity if the value is 0.
	Can handle both scalar floats and numpy arrays.
	"""

	if isinstance( value, numpy.ndarray ):
		to_return = numpy.zeros(( value.shape ))
		to_return[ value > 0 ] = numpy.log( value[ value > 0 ] )
		to_return[ value == 0 ] = NEGINF
		return to_return
	return _log( value )
		
def exp(value):
	"""
	Return e^value, or 0 if the value is - infinity.
	"""
	
	return numpy.exp(value)

def log_probability( model, samples ):
	'''
	Return the log probability of samples given a model.
	'''

	return reduce( lambda x, y: pair_lse( x, y ),
				map( model.log_probability, samples ) )

cdef class Distribution(object):
	"""
	Represents a probability distribution over whatever the HMM you're making is
	supposed to emit. Ought to be subclassed and have log_probability(), 
	sample(), and from_sample() overridden. Distribution.name should be 
	overridden and replaced with a unique name for the distribution type. The 
	distribution should be registered by calling register() on the derived 
	class, so that Distribution.read() can read it. Any distribution parameters 
	need to be floats stored in self.parameters, so they will be properly 
	written by write().
	"""
	
	cdef public str name
	cdef public list parameters, summaries
	cdef public bint frozen

	def __init__( self ):
		"""
		Make a new Distribution with the given parameters. All parameters must 
		be floats.
		
		Storing parameters in self.parameters instead of e.g. self.mean on the 
		one hand makes distribution code ugly, because we don't get to call them
		self.mean. On the other hand, it means we don't have to override the 
		serialization code for every derived class.
		"""

		self.name = "Distribution"
		self.frozen = False
		self.parameters = []
		self.summaries = []

	def __str__( self ):
		"""
		Represent this distribution in a human-readable form.
		"""
		parameters = [ list(p) if isinstance(p, numpy.ndarray) else p
			for p in self.parameters ]
		return "{}({})".format(self.name, ", ".join(map(str, parameters)))

	def __repr__( self ):
		"""
		Represent this distribution in the same format as string.
		"""

		return self.__str__()
		
	def copy( self ):
		"""
		Return a copy of this distribution, untied. 
		"""

		return self.__class__( *self.parameters ) 

	def freeze( self ):
		"""
		Freeze the distribution, preventing training from changing any of the
		parameters of the distribution.
		"""

		self.frozen = True

	def thaw( self ):
		"""
		Thaw the distribution, allowing training to change the parameters of
		the distribution again.
		"""

		self.frozen = False 

	def log_probability( self, symbol ):
		"""
		Return the log probability of the given symbol under this distribution.
		"""
		
		raise NotImplementedError

	def sample( self ):
		"""
		Return a random item sampled from this distribution.
		"""
		
		raise NotImplementedError
		
	def from_sample( self, items, weights=None ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
		
		if self.frozen == True:
			return
		raise NotImplementedError

	def summarize( self, items, weights=None ):
		"""
		Summarize the incoming items into a summary statistic to be used to
		update the parameters upon usage of the `from_summaries` method. By
		default, this will simply store the items and weights into a large
		sample, and call the `from_sample` method.
		"""

		# If no previously stored summaries, just store the incoming data
		if len( self.summaries ) == 0:
			self.summaries = [ items, weights ]

		# Otherwise, append the items and weights
		else:
			prior_items, prior_weights = self.summaries
			items = numpy.concatenate( [prior_items, items] )

			# If even one summary lacks weights, then weights can't be assigned
			# to any of the points.
			if weights is not None:
				weights = numpy.concatenate( [prior_weights, weights] )

			self.summaries = [ items, weights ]

	def from_summaries( self ):
		"""
		Update the parameters of the distribution based on the summaries stored
		previously. 
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		self.from_sample( *self.summaries )
		self.summaries = []

cdef class UniformDistribution( Distribution ):
	"""
	A uniform distribution between two values.
	"""

	def __init__( self, start, end, frozen=False ):
		"""
		Make a new Uniform distribution over floats between start and end, 
		inclusive. Start and end must not be equal.
		"""
		
		# Store the parameters
		self.parameters = [start, end]
		self.summaries = []
		self.name = "UniformDistribution"
		self.frozen = frozen
		
	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""
		
		return self._log_probability( self.parameters[0], self.parameters[1], symbol )

	cdef double _log_probability( self, double a, double b, double symbol ):
		if symbol == a and symbol == b:
			return 0
		if symbol >= a and symbol <= b:
			return _log( 1.0 / ( b - a ) )
		return NEGINF
			
	def sample( self ):
		"""
		Sample from this uniform distribution and return the value sampled.
		"""
		
		return random.uniform(self.parameters[0], self.parameters[1])
		
	def from_sample (self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
		
		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		# Calculate weights. If none are provided, give uniform weights
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return
		
		if len(items) == 0:
			# No sample, so just ignore it and keep our old parameters.
			return
		
		# The ML uniform distribution is just min to max. Weights don't matter
		# for this.
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_min, prior_max = self.parameters
		self.parameters[0] = prior_min*inertia + numpy.min(items)*(1-inertia)
		self.parameters[1] = prior_max*inertia + numpy.max(items)*(1-inertia)

	def summarize( self, items, weights=None ):
		'''
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		'''

		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		if len( items ) == 0:
			# No sample, so just ignore it and keep our own parameters.
			return

		items = numpy.asarray( items )

		# Record the min and max, which are the summary statistics for a
		# uniform distribution.
		self.summaries.append([ items.min(), items.max() ])
		
	def from_summaries( self, inertia=0.0 ):
		'''
		Takes in a series of summaries, consisting of the minimum and maximum
		of a sample, and determine the global minimum and maximum.
		'''

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Load the prior parameters
		prior_min, prior_max = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_min*inertia + summaries[:,0].min()*(1-inertia), 
							prior_max*inertia + summaries[:,1].max()*(1-inertia) ]
		self.summaries = []

cdef class NormalDistribution( Distribution ):
	"""
	A normal distribution based on a mean and standard deviation.
	"""

	def __init__( self, mean, std, frozen=False ):
		"""
		Make a new Normal distribution with the given mean mean and standard 
		deviation std.
		"""
		
		# Store the parameters
		self.parameters = [mean, std]
		self.summaries = []
		self.name = "NormalDistribution"
		self.frozen = frozen

	def log_probability( self, symbol, epsilon=1E-4 ):
		"""
		What's the probability of the given float under this distribution?
		
		For distributions with 0 std, epsilon is the distance within which to 
		consider things equal to the mean.
		"""

		return self._log_probability( symbol, epsilon )

	cdef double _log_probability( self, double symbol, double epsilon ):
		"""
		Do the actual math here.
		"""

		cdef double mu = self.parameters[0], sigma = self.parameters[1]
		if sigma == 0.0:
			if abs( symbol - mu ) < epsilon:
				return 0
			else:
				return NEGINF
  
		return _log( 1.0 / ( sigma * SQRT_2_PI ) ) - ((symbol - mu) ** 2) /\
			(2 * sigma ** 2)

	def sample( self ):
		"""
		Sample from this normal distribution and return the value sampled.
		"""
		
		# This uses the same parameterization
		return random.normalvariate(*self.parameters)
		
	def from_sample( self, items, weights=None, inertia=0.0, min_std=0.01 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		
		min_std specifieds a lower limit on the learned standard deviation.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return
		# The ML uniform distribution is just sample mean and sample std.
		# But we have to weight them. average does weighted mean for us, but 
		# weighted std requires a trick from Stack Overflow.
		# http://stackoverflow.com/a/2415343/402891
		# Take the mean
		mean = numpy.average(items, weights=weights)

		if len(weights[weights != 0]) > 1:
			# We want to do the std too, but only if more than one thing has a 
			# nonzero weight
			# First find the variance
			variance = (numpy.dot(items ** 2 - mean ** 2, weights) / 
				weights.sum())
				
			if variance >= 0:
				std = csqrt(variance)
			else:
				# May have a small negative variance on accident. Ignore and set
				# to 0.
				std = 0
		else:
			# Only one data point, can't update std
			std = self.parameters[1]    
		
		# Enforce min std
		std = max( numpy.array([std, min_std]) )
		
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_mean, prior_std = self.parameters
		self.parameters = [ prior_mean*inertia + mean*(1-inertia), 
							prior_std*inertia + std*(1-inertia) ]

	def summarize( self, items, weights=None ):
		'''
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		'''

		items = numpy.asarray( items )

		# Calculate weights. If none are provided, give uniform weights
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		# Save the mean and variance, the summary statistics for a normal
		# distribution.
		mean = numpy.average( items, weights=weights )
		variance = numpy.dot( items**2 - mean**2, weights ) / weights.sum()

		# Append the mean, variance, and sum of the weights to give the weights
		# of these statistics.
		self.summaries.append( [ mean, variance, weights.sum() ] )
		

	def from_summaries( self, inertia=0.0, min_std=0.01 ):
		'''
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		'''

		# If no summaries stored or the summary is frozen, don't do anything.
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Calculate the new mean and variance. 
		mean = numpy.average( summaries[:,0], weights=summaries[:,2] )
		variance = numpy.sum( [(v+m**2)*w for m, v, w in summaries] ) \
			/ summaries[:,2].sum() - mean**2

		if variance >= 0:
			std = csqrt(variance)
		else:
			std = 0

		std = max( min_std, std )

		# Get the previous parameters.
		prior_mean, prior_std = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_mean*inertia + mean*(1-inertia),
							prior_std*inertia + std*(1-inertia) ]
		self.summaries = []

cdef class LogNormalDistribution( Distribution ):
	"""
	Represents a lognormal distribution over non-negative floats.
	"""

	def __init__( self, mu, sigma, frozen=False ):
		"""
		Make a new lognormal distribution. The parameters are the mu and sigma
		of the normal distribution, which is the the exponential of the log
		normal distribution.
		"""
		self.parameters = [ mu, sigma ]
		self.summaries = []
		self.name = "LogNormalDistribution"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""

		return self._log_probability( symbol )

	cdef double _log_probability( self, symbol ):
		"""
		Actually perform the calculations here, in the Cython-optimized
		function.
		"""

		mu, sigma = self.parameters
		return -clog( symbol * sigma * SQRT_2_PI ) \
			- 0.5 * ( ( clog( symbol ) - mu ) / sigma ) ** 2

	def sample( self ):
		"""
		Return a sample from this distribution.
		"""

		return numpy.random.lognormal( *self.parameters )

	def from_sample( self, items, weights=None, inertia=0.0, min_std=0.01 ):
		"""
		Set the parameters of this distribution to maximize the likelihood of
		the given samples. Items hold some sort of sequence over floats. If
		weights is specified, hold a sequence of values to weight each item by.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		# The ML uniform distribution is just the mean of the log of the samples
		# and sample std the variance of the log of the samples.
		# But we have to weight them. average does weighted mean for us, but 
		# weighted std requires a trick from Stack Overflow.
		# http://stackoverflow.com/a/2415343/402891
		# Take the mean
		mean = numpy.average( numpy.log(items), weights=weights)

		if len(weights[weights != 0]) > 1:
			# We want to do the std too, but only if more than one thing has a 
			# nonzero weight
			# First find the variance
			variance = ( numpy.dot( numpy.log(items) ** 2 - mean ** 2, weights) / 
				weights.sum() )
				
			if variance >= 0:
				std = csqrt(variance)
			else:
				# May have a small negative variance on accident. Ignore and set
				# to 0.
				std = 0
		else:
			# Only one data point, can't update std
			std = self.parameters[1]    
		
		# Enforce min std
		std = max( numpy.array([std, min_std]) )
		
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_mean, prior_std = self.parameters
		self.parameters = [ prior_mean*inertia + mean*(1-inertia), 
							prior_std*inertia + std*(1-inertia) ]

	def summarize( self, items, weights=None ):
		'''
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		'''

		items = numpy.asarray( items )

		# If no weights are specified, use uniform weights.
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		# Calculate the mean and variance, which are the summary statistics
		# for a log-normal distribution.
		mean = numpy.average( numpy.log(items), weights=weights )
		variance = numpy.dot( numpy.log(items)**2 - mean**2, weights ) / weights.sum()
		
		# Save the summary statistics and the weights.
		self.summaries.append( [ mean, variance, weights.sum() ] )
		

	def from_summaries( self, inertia=0.0, min_std=0.01 ):
		'''
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		'''

		# If no summaries are provided or the distribution is frozen, 
		# don't do anything.
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Calculate the mean and variance from the summary statistics.
		mean = numpy.average( summaries[:,0], weights=summaries[:,2] )
		variance = numpy.sum( [(v+m**2)*w for m, v, w in summaries] ) \
			/ summaries[:,2].sum() - mean**2

		if variance >= 0:
			std = csqrt(variance)
		else:
			std = 0

		std = max( min_std, std )

		# Load the previous parameters
		prior_mean, prior_std = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_mean*inertia + mean*(1-inertia), 
							prior_std*inertia + std*(1-inertia) ]
		self.summaries = []

cdef class ExtremeValueDistribution( Distribution ):
	"""
	Represent a generalized extreme value distribution over floats.
	"""

	def __init__( self, mu, sigma, epsilon, frozen=True ):
		"""
		Make a new extreme value distribution, where mu is the location
		parameter, sigma is the scale parameter, and epsilon is the shape
		parameter. 
		"""

		self.parameters = [ float(mu), float(sigma), float(epsilon) ]
		self.name = "ExtremeValueDistribution"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""

		return self._log_probability( symbol )

	cdef double _log_probability( self, symbol ):
		"""
		Actually perform the calculations here, in the Cython-optimized
		function.
		"""

		mu, sigma, epsilon = self.parameters
		t = ( symbol - mu ) / sigma
		if epsilon == 0:
			return -clog( sigma ) - t - cexp( -t )
		return -clog( sigma ) + clog( 1 + epsilon * t ) * (-1. / epsilon - 1) \
			- ( 1 + epsilon * t ) ** ( -1. / epsilon )

cdef class ExponentialDistribution( Distribution ):
	"""
	Represents an exponential distribution on non-negative floats.
	"""
	
	def __init__( self, rate, frozen=False ):
		"""
		Make a new inverse gamma distribution. The parameter is called "rate" 
		because lambda is taken.
		"""

		self.parameters = [rate]
		self.summaries = []
		self.name = "ExponentialDistribution"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""
		
		return _log(self.parameters[0]) - self.parameters[0] * symbol
		
	def sample( self ):
		"""
		Sample from this exponential distribution and return the value
		sampled.
		"""
		
		return random.expovariate(*self.parameters)
		
	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
		
		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return
		
		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return
		
		# Parameter MLE = 1/sample mean, easy to weight
		# Compute the weighted mean
		weighted_mean = numpy.average(items, weights=weights)
		
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_rate = self.parameters[0]
		rate = 1.0 / weighted_mean

		self.parameters[0] = prior_rate*inertia + rate*(1-inertia)

	def summarize( self, items, weights=None ):
		'''
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		'''

		items = numpy.asarray( items )

		# Either store the weights, or assign uniform weights to each item
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		# Calculate the summary statistic, which in this case is the mean.
		mean = numpy.average( items, weights=weights )
		self.summaries.append( [ mean, weights.sum() ] )

	def from_summaries( self, inertia=0.0 ):
		'''
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		'''

		# If no summaries or the distribution is frozen, do nothing.
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Calculate the new parameter from the summary statistics.
		mean = numpy.average( summaries[:,0], weights=summaries[:,1] )

		# Get the parameters
		prior_rate = self.parameters[0]
		rate = 1.0 / mean

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters[0] = prior_rate*inertia + rate*(1-inertia)
		self.summaries = []

cdef class GammaDistribution( Distribution ):
	"""
	This distribution represents a gamma distribution, parameterized in the 
	alpha/beta (shape/rate) parameterization. ML estimation for a gamma 
	distribution, taking into account weights on the data, is nontrivial, and I 
	was unable to find a good theoretical source for how to do it, so I have 
	cobbled together a solution here from less-reputable sources.
	"""
	
	def __init__( self, alpha, beta, frozen=False ):
		"""
		Make a new gamma distribution. Alpha is the shape parameter and beta is 
		the rate parameter.
		"""
		
		self.parameters = [alpha, beta]
		self.summaries = []
		self.name = "GammaDistribution"
		self.frozen = frozen
		
	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""
		
		# Gamma pdf from Wikipedia (and stats class)
		return (_log(self.parameters[1]) * self.parameters[0] - 
			math.lgamma(self.parameters[0]) + 
			_log(symbol) * (self.parameters[0] - 1) - 
			self.parameters[1] * symbol)
		
	def sample( self ):
		"""
		Sample from this gamma distribution and return the value sampled.
		"""
		
		# We have a handy sample from gamma function. Unfortunately, while we 
		# use the alpha, beta parameterization, and this function uses the 
		# alpha, beta parameterization, our alpha/beta are shape/rate, while its
		# alpha/beta are shape/scale. So we have to mess with the parameters.
		return random.gammavariate(self.parameters[0], 1.0 / self.parameters[1])
		
	def from_sample( self, items, weights=None, inertia=0.0, epsilon=1E-9, 
		iteration_limit=1000 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		
		In the Gamma case, likelihood maximization is necesarily numerical, and 
		the extension to weighted values is not trivially obvious. The algorithm
		used here includes a Newton-Raphson step for shape parameter estimation,
		and analytical calculation of the rate parameter. The extension to 
		weights is constructed using vital information found way down at the 
		bottom of an Experts Exchange page.
		
		Newton-Raphson continues until the change in the parameter is less than 
		epsilon, or until iteration_limit is reached
		
		See:
		http://en.wikipedia.org/wiki/Gamma_distribution
		http://www.experts-exchange.com/Other/Math_Science/Q_23943764.html
		"""
		
		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)

		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		# First, do Newton-Raphson for shape parameter.
		
		# Calculate the sufficient statistic s, which is the log of the average 
		# minus the average log. When computing the average log, we weight 
		# outside the log function. (In retrospect, this is actually pretty 
		# obvious.)
		statistic = (log(numpy.average(items, weights=weights)) - 
			numpy.average(log(items), weights=weights))

		# Start our Newton-Raphson at what Wikipedia claims a 1969 paper claims 
		# is a good approximation.
		# Really, start with new_shape set, and shape set to be far away from it
		shape = float("inf")
		
		if statistic != 0:
			# Not going to have a divide by 0 problem here, so use the good
			# estimate
			new_shape =  (3 - statistic + math.sqrt((statistic - 3) ** 2 + 24 * 
				statistic)) / (12 * statistic)
		if statistic == 0 or new_shape <= 0:
			# Try the current shape parameter
			new_shape = self.parameters[0]

		# Count the iterations we take
		iteration = 0
			
		# Now do the update loop.
		# We need the digamma (gamma derivative over gamma) and trigamma 
		# (digamma derivative) functions. Luckily, scipy.special.polygamma(0, x)
		# is the digamma function (0th derivative of the digamma), and 
		# scipy.special.polygamma(1, x) is the trigamma function.
		while abs(shape - new_shape) > epsilon and iteration < iteration_limit:
			shape = new_shape
			
			new_shape = shape - (log(shape) - 
				scipy.special.polygamma(0, shape) -
				statistic) / (1.0 / shape - scipy.special.polygamma(1, shape))
			
			# Don't let shape escape from valid values
			if abs(new_shape) == float("inf") or new_shape == 0:
				# Hack the shape parameter so we don't stop the loop if we land
				# near it.
				shape = new_shape
				
				# Re-start at some random place.
				new_shape = random.random()
				
			iteration += 1
			
		# Might as well grab the new value
		shape = new_shape
				
		# Now our iterative estimation of the shape parameter has converged.
		# Calculate the rate parameter
		rate = 1.0 / (1.0 / (shape * weights.sum()) * items.dot(weights) )

		# Get the previous parameters
		prior_shape, prior_rate = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_shape*inertia + shape*(1-inertia), 
							prior_rate*inertia + rate*(1-inertia) ]    

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		if len(items) == 0:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)

		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		# Save the weighted average of the items, and the weighted average of
		# the log of the items.
		self.summaries.append( [ numpy.average( items, weights=weights ),
								 numpy.average( log(items), weights=weights ),
								 items.dot( weights ),
								 weights.sum() ] )

	def from_summaries( self, inertia=0.0, epsilon=1E-9, 
		iteration_limit=1000 ):
		'''
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample given the summaries which have been stored.
		
		In the Gamma case, likelihood maximization is necesarily numerical, and 
		the extension to weighted values is not trivially obvious. The algorithm
		used here includes a Newton-Raphson step for shape parameter estimation,
		and analytical calculation of the rate parameter. The extension to 
		weights is constructed using vital information found way down at the 
		bottom of an Experts Exchange page.
		
		Newton-Raphson continues until the change in the parameter is less than 
		epsilon, or until iteration_limit is reached

		See:
		http://en.wikipedia.org/wiki/Gamma_distribution
		http://www.experts-exchange.com/Other/Math_Science/Q_23943764.html
		'''

		# If the distribution is frozen, don't bother with any calculation
		if len(self.summaries) == 0 or self.frozen == True:
			return

		# First, do Newton-Raphson for shape parameter.
		
		# Calculate the sufficient statistic s, which is the log of the average 
		# minus the average log. When computing the average log, we weight 
		# outside the log function. (In retrospect, this is actually pretty 
		# obvious.)
		summaries = numpy.array( self.summaries )

		statistic = math.log( numpy.average( summaries[:,0], 
											 weights=summaries[:,3] ) ) - \
					numpy.average( summaries[:,1], 
								   weights=summaries[:,3] )

		# Start our Newton-Raphson at what Wikipedia claims a 1969 paper claims 
		# is a good approximation.
		# Really, start with new_shape set, and shape set to be far away from it
		shape = float("inf")
		
		if statistic != 0:
			# Not going to have a divide by 0 problem here, so use the good
			# estimate
			new_shape =  (3 - statistic + math.sqrt((statistic - 3) ** 2 + 24 * 
				statistic)) / (12 * statistic)
		if statistic == 0 or new_shape <= 0:
			# Try the current shape parameter
			new_shape = self.parameters[0]

		# Count the iterations we take
		iteration = 0
			
		# Now do the update loop.
		# We need the digamma (gamma derivative over gamma) and trigamma 
		# (digamma derivative) functions. Luckily, scipy.special.polygamma(0, x)
		# is the digamma function (0th derivative of the digamma), and 
		# scipy.special.polygamma(1, x) is the trigamma function.
		while abs(shape - new_shape) > epsilon and iteration < iteration_limit:
			shape = new_shape
			
			new_shape = shape - (log(shape) - 
				scipy.special.polygamma(0, shape) -
				statistic) / (1.0 / shape - scipy.special.polygamma(1, shape))
			
			# Don't let shape escape from valid values
			if abs(new_shape) == float("inf") or new_shape == 0:
				# Hack the shape parameter so we don't stop the loop if we land
				# near it.
				shape = new_shape
				
				# Re-start at some random place.
				new_shape = random.random()
				
			iteration += 1
			
		# Might as well grab the new value
		shape = new_shape
				
		# Now our iterative estimation of the shape parameter has converged.
		# Calculate the rate parameter
		rate = 1.0 / (1.0 / (shape * summaries[:,3].sum()) * \
			numpy.sum( summaries[:,2] ) )

		# Get the previous parameters
		prior_shape, prior_rate = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_shape*inertia + shape*(1-inertia), 
							prior_rate*inertia + rate*(1-inertia) ]
		self.summaries = []  		

cdef class InverseGammaDistribution( GammaDistribution ):
	"""
	This distribution represents an inverse gamma distribution (1/the RV ~ gamma
	with the same parameters). A distribution over non-negative floats.
	
	We cheat and don't have to do much work by inheriting from the 
	GammaDistribution.
	"""
	
	def __init__( self, alpha, beta, frozen=False ):
		"""
		Make a new inverse gamma distribution. Alpha is the shape parameter and 
		beta is the scale parameter.
		"""
		
		self.parameters = [alpha, beta]
		self.summaries = []
		self.name = "InverseGammaDistribution"
		self.frozen = frozen
		
	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""
		
		return super(InverseGammaDistribution, self).log_probability(
			1.0 / symbol)
			
	def sample( self ):
		"""
		Sample from this inverse gamma distribution and return the value
		sampled.
		"""
		
		# Invert the sample from the gamma distribution.
		return 1.0 / super( InverseGammaDistribution, self ).sample()
		
	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
		
		# Fit the gamma distribution on the inverted items.
		super( InverseGammaDistribution, self ).from_sample( 1.0 / 
			numpy.asarray( items ), weights=weights, inertia=inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		super( InverseGammaDistribution, self ).summarize( 
			1.0 / numpy.asarray( items ),  weights=weights )

	def from_summaries( self, inertia=0.0, epsilon=1E-9, iteration_limit=1000 ):
		'''
		Update the parameters based on the summaries stored.
		'''

		super( InverseGammaDistribution, self ).from_summaries(
			epsilon=epsilon, iteration_limit=iteration_limit, inertia=inertia )

cdef class DiscreteDistribution(Distribution):
	"""
	A discrete distribution, made up of characters and their probabilities,
	assuming that these probabilities will sum to 1.0. 
	"""
	
	def __init__(self, characters, frozen=False ):
		"""
		Make a new discrete distribution with a dictionary of discrete
		characters and their probabilities, checking to see that these
		sum to 1.0. Each discrete character can be modelled as a
		Bernoulli distribution.
		"""
		
		# Store the parameters
		self.parameters = [ characters ]
		self.summaries = [ {}, 0 ]
		self.name = "DiscreteDistribution"
		self.frozen = frozen


	def log_probability(self, symbol ):
		"""
		What's the probability of the given symbol under this distribution?
		Simply the log probability value given at initiation. If the symbol
		is not part of the discrete distribution, return a log probability
		of NEGINF.
		"""

		return log( self.parameters[0].get( symbol, 0 ) )
			
	def sample( self ):
		"""
		Sample randomly from the discrete distribution, returning the character
		which was randomly generated.
		"""
		
		rand = random.random()
		for key, value in self.parameters[0].items():
			if value >= rand:
				return key
			rand -= value
	
	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Takes in an iterable representing samples from a distribution and
		turn it into a discrete distribution. If no weights are provided,
		each sample is weighted equally. If weights are provided, they are
		normalized to sum to 1 and used.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len( items ) == 0 or self.frozen == True:
			return

		n = len( items )

		# Normalize weights, or assign uniform probabilities
		if weights is None:
			weights = numpy.ones( n ) / n
		else:
			weights = numpy.array(weights) / numpy.sum(weights)

		# Sum the weights seen for each character
		characters = {}
		for character, weight in izip( items, weights ):
			try:
				characters[character] += weight
			except KeyError:
				characters[character] = weight

		# Adjust the new weights by the inertia
		for character, weight in characters.items():
			characters[character] = weight * (1-inertia)

		# Adjust the old weights by the inertia
		prior_characters = self.parameters[0]
		for character, weight in prior_characters.items():
			try:
				characters[character] += weight * inertia
			except KeyError:
				characters[character] = weight * inertia

		self.parameters = [ characters ]

	def summarize( self, items, weights=None ):
		'''
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		'''

		n = len( items )
		if weights is None:
			weights = numpy.ones( n )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		characters = self.summaries[0]
		for character, weight in izip( items, weights ):
			try:
				characters[character] += weight
			except KeyError:
				characters[character] = weight

		self.summaries[0] = characters
		self.summaries[1] += weights.sum()

	def from_summaries( self, inertia=0.0 ):
		'''
		Takes in a series of summaries and merge them.
		'''

		# If the distribution is frozen, don't bother with any calculation
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		# Unpack the variables
		prior_characters = self.parameters[0]
		characters, total_weight = self.summaries 

		# Scale the characters by both the total number of weights and by
		# the inertia.
		for character, prob in characters.items():
			characters[character] = ( prob / total_weight ) * (1-inertia)

		# Adjust the old weights by the inertia
		if inertia > 0.0:
			for character, weight in prior_characters.items():
				try:
					characters[character] += weight * inertia
				except KeyError:
					characters[character] = weight * inertia

		self.parameters = [ characters ]
		self.summaries = [ {}, 0 ]


cdef class LambdaDistribution(Distribution):
	"""
	A distribution which takes in an arbitrary lambda function, and returns
	probabilities associated with whatever that function gives. For example...

	func = lambda x: log(1) if 2 > x > 1 else log(0)
	distribution = LambdaDistribution( func )
	print distribution.log_probability( 1 ) # 1
	print distribution.log_probability( -100 ) # 0

	This assumes the lambda function returns the log probability, not the
	untransformed probability.
	"""
	
	def __init__(self, lambda_funct, frozen=True ):
		"""
		Takes in a lambda function and stores it. This function should return
		the log probability of seeing a certain input.
		"""

		# Store the parameters
		self.parameters = [lambda_funct]
		self.name = "LambdaDistribution"
		self.frozen = frozen
		
	def log_probability(self, symbol):
		"""
		What's the probability of the given float under this distribution?
		"""

		return self.parameters[0](symbol)

cdef class GaussianKernelDensity( Distribution ):
	"""
	A quick way of storing points to represent a Gaussian kernel density in one
	dimension. Takes in the points at initialization, and calculates the log of
	the sum of the Gaussian distance of the new point from every other point.
	"""

	def __init__( self, points, bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		points = numpy.asarray( points )
		n = len(points)
		
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		self.parameters = [ points, bandwidth, weights ]
		self.summaries = []
		self.name = "GaussianKernelDensity"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of a given float under this distribution? It's
		the sum of the distances of the symbol from every point stored in the
		density. Bandwidth is defined at the beginning. A wrapper for the
		cython function which does math.
		"""

		return self._log_probability( symbol )

	cdef double _log_probability( self, double symbol ):
		"""
		Actually calculate it here.
		"""
		cdef double bandwidth = self.parameters[1]
		cdef double mu, scalar = 1.0 / SQRT_2_PI
		cdef int i = 0, n = len(self.parameters[0])
		cdef double distribution_prob = 0, point_prob

		for i in xrange( n ):
			# Go through each point sequentially
			mu = self.parameters[0][i]

			# Calculate the probability under that point
			point_prob = scalar * \
				cexp( -0.5 * (( mu-symbol ) / bandwidth) ** 2 )

			# Scale that point according to the weight 
			distribution_prob += point_prob * self.parameters[2][i]

		# Return the log of the sum of the probabilities
		return _log( distribution_prob )

	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		return random.gauss( mu, self.parameters[1] )

	def from_sample( self, points, weights=None, inertia=0.0 ):
		"""
		Replace the points, allowing for inertia if specified.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		points = numpy.asarray( points )
		n = len(points)

		# Get the weights, or assign uniform weights
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		# If no inertia, get rid of the previous points
		if inertia == 0.0:
			self.parameters[0] = points
			self.parameters[2] = weights

		# Otherwise adjust weights appropriately
		else: 
			self.parameters[0] = numpy.concatenate( ( self.parameters[0],
													  points ) )
			self.parameters[2] = numpy.concatenate( ( self.parameters[2]*inertia,
													  weights*(1-inertia) ) )

cdef class UniformKernelDensity( Distribution ):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __init__( self, points, bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		points = numpy.asarray( points )
		n = len(points)
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		self.parameters = [ points, bandwidth, weights ]
		self.summaries = []
		self.name = "UniformKernelDensity"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability ofa given float under this distribution? It's
		the sum of the distances from the symbol calculated under individual
		exponential distributions. A wrapper for the cython function.
		"""

		return self._log_probability( symbol )

	cdef _log_probability( self, double symbol ):
		"""
		Actually do math here.
		"""

		cdef double mu
		cdef double distribution_prob=0, point_prob
		cdef int i = 0, n = len(self.parameters[0])

		for i in xrange( n ):
			# Go through each point sequentially
			mu = self.parameters[0][i]

			# The good thing about uniform distributions if that
			# you just need to check to make sure the point is within
			# a bandwidth.
			if abs( mu - symbol ) <= self.parameters[1]:
				point_prob = 1
			else:
				point_prob = 0

			# Properly weight the point before adding it to the sum
			distribution_prob += point_prob * self.parameters[2][i]

		# Return the log of the sum of probabilities
		return _log( distribution_prob )
	
	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		bandwidth = self.parameters[1]
		return random.uniform( mu-bandwidth, mu+bandwidth )

	def from_sample( self, points, weights=None, inertia=0.0 ):
		"""
		Replace the points, allowing for inertia if specified.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		points = numpy.asarray( points )
		n = len(points)

		# Get the weights, or assign uniform weights
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		# If no inertia, get rid of the previous points
		if inertia == 0.0:
			self.parameters[0] = points
			self.parameters[2] = weights

		# Otherwise adjust weights appropriately
		else: 
			self.parameters[0] = numpy.concatenate( ( self.parameters[0],
													  points ) )
			self.parameters[2] = numpy.concatenate( ( self.parameters[2]*inertia,
													  weights*(1-inertia) ) )

cdef class TriangleKernelDensity( Distribution ):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __init__( self, points, bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		points = numpy.asarray( points )
		n = len(points)
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		self.parameters = [ points, bandwidth, weights ]
		self.summaries = []
		self.name = "TriangleKernelDensity"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of a given float under this distribution? It's
		the sum of the distances from the symbol calculated under individual
		exponential distributions. A wrapper for the cython function.
		""" 

		return self._log_probability( symbol )

	cdef double _log_probability( self, double symbol ):
		"""
		Actually do math here.
		"""

		cdef double bandwidth = self.parameters[1]
		cdef double mu
		cdef double distribution_prob=0, point_prob
		cdef int i = 0, n = len(self.parameters[0])

		for i in xrange( n ):
			# Go through each point sequentially
			mu = self.parameters[0][i]

			# Calculate the probability for each point
			point_prob = bandwidth - abs( mu - symbol ) 
			if point_prob < 0:
				point_prob = 0 

			# Properly weight the point before adding to the sum
			distribution_prob += point_prob * self.parameters[2][i]

		# Return the log of the sum of probabilities
		return _log( distribution_prob )

	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		bandwidth = self.parameters[1]
		return random.triangular( mu-bandwidth, mu+bandwidth, mu )

	def from_sample( self, points, weights=None, inertia=0.0 ):
		"""
		Replace the points, allowing for inertia if specified.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		points = numpy.asarray( points )
		n = len(points)

		# Get the weights, or assign uniform weights
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		# If no inertia, get rid of the previous points
		if inertia == 0.0:
			self.parameters[0] = points
			self.parameters[2] = weights

		# Otherwise adjust weights appropriately
		else: 
			self.parameters[0] = numpy.concatenate( ( self.parameters[0],
													  points ) )
			self.parameters[2] = numpy.concatenate( ( self.parameters[2]*inertia,
													  weights*(1-inertia) ) )

cdef class MixtureDistribution( Distribution ):
	"""
	Allows you to create an arbitrary mixture of distributions. There can be
	any number of distributions, include any permutation of types of
	distributions. Can also specify weights for the distributions.
	"""

	def __init__( self, distributions, weights=None, frozen=False ):
		"""
		Take in the distributions and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""
		n = len(distributions)
		if weights:
			weights = numpy.array( weights ) / numpy.sum( weights )
		else:
			weights = numpy.ones(n) / n

		self.parameters = [ distributions, weights ]
		self.summaries = []
		self.name = "MixtureDistribution"
		self.frozen = frozen

	def __str__( self ):
		"""
		Return a string representation of this mixture.
		"""

		distributions, weights = self.parameters
		distributions = map( str, distributions )
		return "MixtureDistribution( {}, {} )".format(
			distributions, list(weights) ).replace( "'", "" )

	def log_probability( self, symbol ):
		"""
		What's the probability of a given float under this mixture? It's
		the log-sum-exp of the distances from the symbol calculated under all
		distributions. Currently in python, not cython, to allow for dovetyping
		of both numeric and not-necessarily-numeric distributions. 
		"""

		(d, w), n = self.parameters, len(self.parameters)
		return _log( numpy.sum([ cexp( d[i].log_probability(symbol) ) \
			* w[i] for i in xrange( len(d) ) ]) )

	def sample( self ):
		"""
		Sample from the mixture. First, choose a distribution to sample from
		according to the weights, then sample from that distribution. 
		"""

		i = random.random()
		for d, w in zip( *self.parameters ):
			if w > i:
				return d.sample()
			i -= w 

	def from_sample( self, items, weights=None ):
		"""
		Perform EM to estimate the parameters of each distribution
		which is a part of this mixture.
		"""

		if weights is None:
			weights = numpy.ones( len(items) )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		distributions, w = self.parameters
		n, k = len(items), len(distributions)

		# The responsibility matrix
		r = numpy.zeros( (n, k) )

		# Calculate the log probabilities of each p
		for i, distribution in enumerate( distributions ):
			for j, item in enumerate( items ):
				r[j, i] = distribution.log_probability( item )

		r = numpy.exp( r )

		# Turn these log probabilities into responsibilities by
		# normalizing on a row-by-row manner.
		for i in xrange( n ):
			r[i] = r[i] / r[i].sum()

		# Weight the responsibilities by the given weights
		for i in xrange( k ):
			r[:,i] = r[:,i]*weights

		# Update the emissions of each distribution
		for i, distribution in enumerate( distributions ):
			distribution.from_sample( items, weights=r[:,i] )

		# Update the weight of each distribution
		self.parameters[1] = r.sum( axis=0 ) / r.sum()

	def summarize( self, items, weights=None ):
		"""
		Performs the summary step of the EM algorithm to estimate
		parameters of each distribution which is a part of this mixture.
		"""

		if weights is None:
			weights = numpy.ones( len(items) )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		distributions, w = self.parameters
		n, k = len(items), len(distributions)

		# The responsibility matrix
		r = numpy.zeros( (n, k) )

		# Calculate the log probabilities of each p
		for i, distribution in enumerate( distributions ):
			for j, item in enumerate( items ):
				r[j, i] = distribution.log_probability( item )

		r = numpy.exp( r )

		# Turn these log probabilities into responsibilities by
		# normalizing on a row-by-row manner.
		for i in xrange( n ):
			r[i] = r[i] / r[i].sum()

		# Weight the responsibilities by the given weights
		for i in xrange( k ):
			r[:,i] = r[:,i]*weights

		# Save summary statistics on the emission distributions
		for i, distribution in enumerate( distributions ):
			distribution.summarize( items, weights=r[:,i]*weights )

		# Save summary statistics for weight updates
		self.summaries.append( r.sum( axis=0 ) / r.sum() )

	def from_summaries( self, inertia=0.0 ):
		"""
		Performs the actual update step for the EM algorithm.
		"""

		# If this distribution is frozen, don't do anything.
		if self.frozen == True:
			return

		# Update the emission distributions
		for d in self.parameters[0]:
			d.from_summaries( inertia=inertia )

		# Update the weights
		weights = numpy.array( self.summaries )
		self.parameters[1] = weights.sum( axis=0 ) / weights.sum()

cdef class MultivariateDistribution( Distribution ):
	"""
	Allows you to create a multivariate distribution, where each distribution
	is independent of the others. Distributions can be any type, such as
	having an exponential represent the duration of an event, and a normal
	represent the mean of that event. Observations must now be tuples of
	a length equal to the number of distributions passed in.

	s1 = MultivariateDistribution([ ExponentialDistribution( 0.1 ), 
									NormalDistribution( 5, 2 ) ])
	s1.log_probability( (5, 2 ) )
	"""

	def __init__( self, distributions, weights=None, frozen=False ):
		"""
		Take in the distributions and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""
		n = len(distributions)
		if weights:
			weights = numpy.array( weights )
		else:
			weights = numpy.ones(n)

		self.parameters = [ distributions, weights ]
		self.name = "MultivariateDistribution"
		self.frozen = frozen

	def __str__( self ):
		"""
		Return a string representation of the MultivariateDistribution.
		"""

		distributions = map( str, self.parameters[0] )
		return "MultivariateDistribution({})".format(
			distributions ).replace( "'", "" )

	def log_probability( self, symbol ):
		"""
		What's the probability of a given tuple under this mixture? It's the
		product of the probabilities of each symbol in the tuple under their
		respective distribution, which is the sum of the log probabilities.
		"""

		return sum( d.log_probability( obs )*w for d, obs, w in zip( 
			self.parameters[0], symbol, self.parameters[1] ) )

	def sample( self ):
		"""
		Sample from the mixture. First, choose a distribution to sample from
		according to the weights, then sample from that distribution. 
		"""

		return [ d.sample() for d in self.parameters[0] ]

	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Items are tuples, and so each distribution can be trained
		independently of each other. 
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		items = numpy.asarray( items )

		for i, d in enumerate( self.parameters[0] ):
			d.from_sample( items[:,i], weights=weights, inertia=inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in an array of items and reduce it down to summary statistics. For
		a multivariate distribution, this involves just passing the appropriate
		data down to the appropriate distributions.
		"""

		items = numpy.asarray( items )

		for i, d in enumerate( self.parameters[0] ):
			d.summarize( items[:,i], weights=weights )

	def from_summaries( self, inertia=0.0 ):
		"""
		Use the collected summary statistics in order to update the
		distributions.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		for d in self.parameters[0]:
			d.from_summaries( inertia=inertia )


cdef class State(object):
	"""
	Represents a state in an HMM. Holds emission distribution, but not
	transition distribution, because that's stored in the graph edges.
	"""
	
	cdef public Distribution distribution
	cdef public str name
	cdef public str identity
	cdef public double weight

	def __init__(self, distribution, name=None, weight=None, identity=None ):
		"""
		Make a new State emitting from the given distribution. If distribution 
		is None, this state does not emit anything. A name, if specified, will 
		be the state's name when presented in output. Name may not contain 
		spaces or newlines, and must be unique within an HMM. Identity is a
		store of the id property, to allow for multiple states to have the same
		name but be uniquely identifiable. 
		"""
		
		# Save the distribution
		self.distribution = distribution
		
		# Save the name
		self.name = name or str(id(str))

		# Save the id
		if identity is not None:
			self.identity = str(identity)
		else:
			self.identity = str(id(self))

		self.weight = weight or 1.

	def __str__(self):
		"""
		Represent this state with it's name, weight, and identity.
		"""
		
		return "State( {}, name={}, weight={}, identity={} )".format(
			str(self.distribution), self.name, self.weight, self.identity )

	def is_silent(self):
		"""
		Return True if this state is silent (distribution is None) and False 
		otherwise.
		"""
		
		return self.distribution is None

	def tied_copy( self ):
		"""
		Return a copy of this state where the distribution is tied to the
		distribution of this state.
		"""

		return State( distribution=self.distribution, name=self.name )
		
	def copy( self ):
		"""
		Return a hard copy of this state.
		"""

		return State( **self.__dict__ )
		
	def write(self, stream):
		"""
		Write this State (and its Distribution) to the given stream.
		
		Format: name, followed by "*" if the state is silent.
		If not followed by "*", the next line contains the emission
		distribution.
		"""
		
		name = self.name.replace( " ", "_" ) 
		stream.write( "{} {} {} {}\n".format( 
			self.identity, name, self.weight, str( self.distribution ) ) )
		
	@classmethod
	def read(cls, stream):
		"""
		Read a State from the given stream, in the format output by write().
		"""
		
		# Read a line
		line = stream.readline()
		
		if line == "":
			raise EOFError("End of file while reading state.")
			
		# Spilt the line up
		parts = line.strip().split()
		
		# parts[0] holds the state's name, and parts[1] holds the rest of the
		# state information, so we can just evaluate it.
		identity, name, weight, distribution = \
			parts[0], parts[1], parts[2], ' '.join( parts[3:] )
		return eval( "State( {}, name='{}', weight={}, identity='{}' )".format( 
			distribution, name, weight, identity ) )

cdef class Model(object):
	"""
	Represents a Hidden Markov Model.
	"""
	cdef public str name
	cdef public object start, end, graph
	cdef public list states
	cdef public int start_index, end_index, silent_start
	cdef int [:] in_edge_count, in_transitions, out_edge_count, out_transitions
	cdef double [:] in_transition_log_probabilities
	cdef double [:] in_transition_pseudocounts
	cdef double [:] out_transition_log_probabilities
	cdef double [:] out_transition_pseudocounts
	cdef double [:] state_weights
	cdef int [:] tied_state_count
	cdef int [:] tied
	cdef int [:] tied_edge_group_size
	cdef int [:] tied_edges_starts
	cdef int [:] tied_edges_ends
	cdef int finite

	def __init__(self, name=None, start=None, end=None):
		"""
		Make a new Hidden Markov Model. Name is an optional string used to name
		the model when output. Name may not contain spaces or newlines.
		
		If start and end are specified, they are used as start and end states 
		and new start and end states are not generated.
		"""
		
		# Save the name or make up a name.
		self.name = name or str( id(self) )

		# This holds a directed graph between states. Nodes in that graph are
		# State objects, so they're guaranteed never to conflict when composing
		# two distinct models
		self.graph = networkx.DiGraph()
		
		# Save the start or make up a start
		self.start = start or State( None, name=self.name + "-start" )

		# Save the end or make up a end
		self.end = end or State( None, name=self.name + "-end" )
		
		# Put start and end in the graph
		self.graph.add_node(self.start)
		self.graph.add_node(self.end)
	
	def __str__(self):
		"""
		Represent this HMM with it's name and states.
		"""
		
		return "{}:\n\t{}".format(self.name, "\n\t".join(map(str, self.states)))

	def state_count( self ):
		"""
		Returns the number of states present in the model.
		"""

		return len( self.states )

	def edge_count( self ):
		"""
		Returns the number of edges present in the model.
		"""

		return len( self.out_transition_log_probabilities )

	def dense_transition_matrix( self ):
		"""
		Returns the dense transition matrix. Useful if the transitions of
		somewhat small models need to be analyzed.
		"""

		m = len(self.states)
		transition_log_probabilities = numpy.zeros( (m, m) ) + NEGINF

		for i in xrange(m):
			for n in xrange( self.out_edge_count[i], self.out_edge_count[i+1] ):
				transition_log_probabilities[i, self.out_transitions[n]] = \
					self.out_transition_log_probabilities[n]

		return transition_log_probabilities 

	def is_infinite( self ):
		"""
		Returns whether or not the HMM is infinite, or finite. An infinite HMM
		is a HMM which does not have explicit transitions to an end state,
		meaning that it can end in any symbol emitting state. This is
		determined in the bake method, based on if there are any edges to the
		end state or not. Can only be used after a model is baked.
		"""

		return self.finite == 0

	def add_state(self, state):
		"""
		Adds the given State to the model. It must not already be in the model,
		nor may it be part of any other model that will eventually be combined
		with this one.
		"""
		
		# Put it in the graph
		self.graph.add_node(state)

	def add_states( self, states ):
		"""
		Adds multiple states to the model at the same time. Basically just a
		helper function for the add_state method.
		"""

		for state in states:
			self.add_state( state )
		
	def add_transition( self, a, b, probability, pseudocount=None, group=None ):
		"""
		Add a transition from state a to state b with the given (non-log)
		probability. Both states must be in the HMM already. self.start and
		self.end are valid arguments here. Probabilities will be normalized
		such that every node has edges summing to 1. leaving that node, but
		only when the model is baked. 

		By specifying a group as a string, you can tie edges together by giving
		them the same group. This means that a transition across one edge in the
		group counts as a transition across all edges in terms of training.
		"""
		
		# If a pseudocount is specified, use it, otherwise use the probability.
		# The pseudocounts come up during training, when you want to specify
		# custom pseudocount weighting schemes per edge, in order to make the
		# model converge to that scheme given no observations. 
		pseudocount = pseudocount or probability

		# Add the transition
		self.graph.add_edge(a, b, weight=log(probability), 
			pseudocount=pseudocount, group=group )

	def add_transitions( self, a, b, probabilities=None, pseudocounts=None,
		groups=None ):
		"""
		Add many transitions at the same time, in one of two forms. 

		(1) If both a and b are lists, then create transitions from the i-th 
		element of a to the i-th element of b with a probability equal to the
		i-th element of probabilities.

		Example: 
		model.add_transitions([model.start, s1], [s1, model.end], [1., 1.])

		(2) If either a or b are a state, and the other is a list, create a
		transition from all states in the list to the single state object with
		probabilities and pseudocounts specified appropriately.

		Example:
		model.add_transitions([model.start, s1, s2, s3], s4, [0.2, 0.4, 0.3, 0.9])
		model.add_transitions(model.start, [s1, s2, s3], [0.6, 0.2, 0.05])

		If a single group is given, it's assumed all edges should belong to that
		group. Otherwise, either groups can be a list of group identities, or
		simply None if no group is meant.
		"""

		# If a pseudocount is specified, use it, otherwise use the probability.
		# The pseudocounts come up during training, when you want to specify
		# custom pseudocount weighting schemes per edge, in order to make the
		# model converge to that scheme given no observations. 
		pseudocounts = pseudocounts or probabilities

		n = len(a) if isinstance( a, list ) else len(b)
		if groups is None or isinstance( groups, str ):
			groups = [ groups ] * n

		# Allow addition of many transitions from many states
		if isinstance( a, list ) and isinstance( b, list ):
			# Set up an iterator across all edges
			edges = izip( a, b, probabilities, pseudocounts, groups )
			
			for start, end, probability, pseudocount, group in edges:
				self.add_transition( start, end, probability, pseudocount, group )

		# Allow for multiple transitions to a specific state 
		elif isinstance( a, list ) and isinstance( b, State ):
			# Set up an iterator across all edges to b
			edges = izip( a, probabilities, pseudocounts, groups )

			for start, probability, pseudocount, group in edges:
				self.add_transition( start, b, probability, pseudocount, group )

		# Allow for multiple transitions from a specific state
		elif isinstance( a, State ) and isinstance( b, list ):
			# Set up an iterator across all edges from a
			edges = izip( b, probabilities, pseudocounts, groups )

			for end, probability, pseudocount, group in edges:
				self.add_transition( a, end, probability, pseudocount, group )

	def add_model( self, other ):
		"""
		Given another Model, add that model's contents to us. Its start and end
		states become silent states in our model.
		"""
		
		# Unify the graphs (requiring disjoint states)
		self.graph = networkx.union(self.graph, other.graph)
		
		# Since the nodes in the graph are references to Python objects,
		# other.start and other.end and self.start and self.end still mean the
		# same State objects in the new combined graph.

	def concatenate_model( self, other ):
		"""
		Given another model, concatenate it in such a manner that you simply
		add a transition of probability 1 from self.end to other.start, and
		set the end of this model to other.end.
		"""

		# Unify the graphs (requiring disjoint states)
		self.graph = networkx.union( self.graph, other.graph )
		
		# Connect the two graphs
		self.add_transition( self.end, other.start, 1.00 )

		# Move the end to other.end
		self.end = other.end

	def draw( self, **kwargs ):
		"""
		Draw this model's graph using NetworkX and matplotlib. Blocks until the
		window displaying the graph is closed.
		
		Note that this relies on networkx's built-in graphing capabilities (and 
		not Graphviz) and thus can't draw self-loops.

		See networkx.draw_networkx() for the keywords you can pass in.
		"""
		
		networkx.draw(self.graph, **kwargs)
		pyplot.show()

	def freeze_distributions( self ):
		"""
		Freeze all the distributions in model. This means that upon training,
		only edges will be updated. The parameters of distributions will not
		be affected.
		"""

		for state in self.states:
			state.distribution.freeze()

	def thaw_distributions( self ):
		"""
		Thaw all distributions in the model. This means that upon training,
		distributions will be updated again.
		"""

		for state in self.states:
			state.distribution.thaw()

	def bake( self, verbose=False, merge="all" ): 
		"""
		Finalize the topology of the model, and assign a numerical index to
		every state. This method must be called before any of the probability-
		calculating methods.
		
		This fills in self.states (a list of all states in order) and 
		self.transition_log_probabilities (log probabilities for transitions), 
		as well as self.start_index and self.end_index, and self.silent_start 
		(the index of the first silent state).

		The option verbose will return a log of the changes made to the model
		due to normalization or merging. 

		Merging has three options:
			"None": No modifications will be made to the model.
			"Partial": A silent state which only has a probability 1 transition
				to another silent state will be merged with that silent state.
				This means that if silent state "S1" has a single transition
				to silent state "S2", that all transitions to S1 will now go
				to S2, with the same probability as before, and S1 will be
				removed from the model.
			"All": A silent state with a probability 1 transition to any other
				state, silent or symbol emitting, will be merged in the manner
				described above. In addition, any orphan states will be removed
				from the model. An orphan state is a state which does not have
				any transitions to it OR does not have any transitions from it,
				except for the start and end of the model. This will iteratively
				remove orphan chains from the model. This is sometimes desirable,
				as all states should have both a transition in to get to that
				state, and a transition out, even if it is only to itself. If
				the state does not have either, the HMM will likely not work as
				intended.
		"""

		# Go through the model and delete any nodes which have no edges leading
		# to it, or edges leading out of it. This gets rid of any states with
		# no edges in or out, as well as recursively removing any chains which
		# are impossible for the viterbi path to touch.
		self.in_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 ) 
		self.out_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 )
		
		merge = merge.lower() if merge else None
		while merge == 'all':
			merge_count = 0

			# Reindex the states based on ones which are still there
			prestates = self.graph.nodes()
			indices = { prestates[i]: i for i in xrange( len( prestates ) ) }

			# Go through all the edges, summing in and out edges
			for a, b in self.graph.edges():
				self.out_edge_count[ indices[a] ] += 1
				self.in_edge_count[ indices[b] ] += 1
				
			# Go through each state, and if either in or out edges are 0,
			# remove the edge.
			for i in xrange( len( prestates ) ):
				if prestates[i] is self.start or prestates[i] is self.end:
					continue

				if self.in_edge_count[i] == 0:
					merge_count += 1
					self.graph.remove_node( prestates[i] )

					if verbose:
						print "Orphan state {} removed due to no edges \
							leading to it".format(prestates[i].name )

				elif self.out_edge_count[i] == 0:
					merge_count += 1
					self.graph.remove_node( prestates[i] )

					if verbose:
						print "Orphan state {} removed due to no edges \
							leaving it".format(prestates[i].name )

			if merge_count == 0:
				break

		# Go through the model checking to make sure out edges sum to 1.
		# Normalize them to 1 if this is not the case.
		for state in self.graph.nodes():

			# Perform log sum exp on the edges to see if they properly sum to 1
			out_edges = round( sum( numpy.e**x['weight'] 
				for x in self.graph.edge[state].values() ), 8 )

			# The end state has no out edges, so will be 0
			if out_edges != 1. and state != self.end:
				# Issue a notice if verbose is activated
				if verbose:
					print "{} : {} summed to {}, normalized to 1.0"\
						.format( self.name, state.name, out_edges )

				# Reweight the edges so that the probability (not logp) sums
				# to 1.
				for edge in self.graph.edge[state].values():
					edge['weight'] = edge['weight'] - log( out_edges )

		# Automatically merge adjacent silent states attached by a single edge
		# of 1.0 probability, as that adds nothing to the model. Traverse the
		# edges looking for 1.0 probability edges between silent states.
		while merge in ['all', 'partial']:
			# Repeatedly go through the model until no merges take place.
			merge_count = 0

			for a, b, e in self.graph.edges( data=True ):
				# Since we may have removed a or b in a previous iteration,
				# a simple fix is to just check to see if it's still there
				if a not in self.graph.nodes() or b not in self.graph.nodes():
					continue

				if a == self.start or b == self.end:
					continue

				# If a silent state has a probability 1 transition out
				if e['weight'] == 0.0 and a.is_silent():

					# Make sure the transition is an appropriate merger
					if merge=='all' or ( merge=='partial' and b.is_silent() ):

						# Go through every transition to that state 
						for x, y, d in self.graph.edges( data=True ):

							# Make sure that the edge points to the current node
							if y is a:
								# Increment the edge counter
								merge_count += 1

								# Remove the edge going to that node
								self.graph.remove_edge( x, y )

								pseudo = max( e['pseudocount'], d['pseudocount'] )
								group = e['group'] if e['group'] == d['group'] else None
								# Add a new edge going to the new node
								self.graph.add_edge( x, b, weight=d['weight'],
									pseudocount=pseudo,
									group=group )

								# Log the event
								if verbose:
									print "{} : {} - {} merged".format(
										self.name, a, b)

						# Remove the state now that all edges are removed
						self.graph.remove_node( a )

			if merge_count == 0:
				break

		# Detect whether or not there are loops of silent states by going
		# through every pair of edges, and ensure that there is not a cycle
		# of silent states.		
		for a, b, e in self.graph.edges( data=True ):
			for x, y, d in self.graph.edges( data=True ):
				if a is y and b is x and a.is_silent() and b.is_silent():
					print "Loop: {} - {}".format( a.name, b.name )

		states = self.graph.nodes()
		n, m = len(states), len(self.graph.edges())
		silent_states, normal_states = [], []

		for state in states:
			if state.is_silent():
				silent_states.append(state)
			else:
				normal_states.append(state)

		# We need the silent states to be in topological sort order: any
		# transition between silent states must be from a lower-numbered state
		# to a higher-numbered state. Since we ban loops of silent states, we
		# can get away with this.
		
		# Get the subgraph of all silent states
		silent_subgraph = self.graph.subgraph(silent_states)
		
		# Get the sorted silent states. Isn't it convenient how NetworkX has
		# exactly the algorithm we need?
		silent_states_sorted = networkx.topological_sort(silent_subgraph)
		
		# What's the index of the first silent state?
		self.silent_start = len(normal_states)

		# Save the master state ordering. Silent states are last and in
		# topological order, so when calculationg forward algorithm
		# probabilities we can just go down the list of states.
		self.states = normal_states + silent_states_sorted 
		
		# We need a good way to get transition probabilities by state index that
		# isn't N^2 to build or store. So we will need a reverse of the above
		# mapping. It's awkward but asymptotically fine.
		indices = { self.states[i]: i for i in xrange(n) }

		# Create a sparse representation of the tied states in the model. This
		# is done in the same way of the transition, by having a vector of
		# counts, and a vector of the IDs that the state is tied to.
		self.tied_state_count = numpy.zeros( self.silent_start+1, 
			dtype=numpy.int32 )

		for i in xrange( self.silent_start ):
			for j in xrange( self.silent_start ):
				if i == j:
					continue
				if self.states[i].distribution is self.states[j].distribution:
					self.tied_state_count[i+1] += 1

		# Take the cumulative sum in order to get indexes instead of counts,
		# with the last index being the total number of ties.
		self.tied_state_count = numpy.cumsum( self.tied_state_count,
			dtype=numpy.int32 )

		self.tied = numpy.zeros( self.tied_state_count[-1], 
			dtype=numpy.int32 ) - 1

		for i in xrange( self.silent_start ):
			for j in xrange( self.silent_start ):
				if i == j:
					continue
					
				if self.states[i].distribution is self.states[j].distribution:
					# Begin at the first index which belongs to state i...
					start = self.tied_state_count[i]

					# Find the first non -1 entry in order to put our index.
					while self.tied[start] != -1:
						start += 1

					# Now that we've found a non -1 entry, put the index of the
					# state which this state is tied to in!
					self.tied[start] = j

		# Unpack the state weights
		self.state_weights = numpy.zeros( self.silent_start )
		for i in xrange( self.silent_start ):
			self.state_weights[i] = clog( self.states[i].weight )

		# This holds numpy array indexed [a, b] to transition log probabilities 
		# from a to b, where a and b are state indices. It starts out saying all
		# transitions are impossible.
		self.in_transitions = numpy.zeros( len(self.graph.edges()), 
			dtype=numpy.int32 ) - 1
		self.in_edge_count = numpy.zeros( len(self.states)+1, 
			dtype=numpy.int32 ) 
		self.out_transitions = numpy.zeros( len(self.graph.edges()), 
			dtype=numpy.int32 ) - 1
		self.out_edge_count = numpy.zeros( len(self.states)+1, 
			dtype=numpy.int32 )
		self.in_transition_log_probabilities = numpy.zeros(
			len( self.graph.edges() ) )
		self.out_transition_log_probabilities = numpy.zeros(
			len( self.graph.edges() ) )
		self.in_transition_pseudocounts = numpy.zeros( 
			len( self.graph.edges() ) )
		self.out_transition_pseudocounts = numpy.zeros(
			len( self.graph.edges() ) )

		# Now we need to find a way of storing in-edges for a state in a manner
		# that can be called in the cythonized methods below. This is basically
		# an inversion of the graph. We will do this by having two lists, one
		# list size number of nodes + 1, and one list size number of edges.
		# The node size list will store the beginning and end values in the
		# edge list that point to that node. The edge list will be ordered in
		# such a manner that all edges pointing to the same node are grouped
		# together. This will allow us to run the algorithms in time
		# nodes*edges instead of nodes*nodes.

		for a, b in self.graph.edges_iter():
			# Increment the total number of edges going to node b.
			self.in_edge_count[ indices[b]+1 ] += 1
			# Increment the total number of edges leaving node a.
			self.out_edge_count[ indices[a]+1 ] += 1

		# Determine if the model is infinite or not based on the number of edges
		# to the end state
		if self.in_edge_count[ indices[ self.end ]+1 ] == 0:
			self.finite = 0
		else:
			self.finite = 1

		# Take the cumulative sum so that we can associate array indices with
		# in or out transitions
		self.in_edge_count = numpy.cumsum(self.in_edge_count, 
			dtype=numpy.int32)
		self.out_edge_count = numpy.cumsum(self.out_edge_count, 
			dtype=numpy.int32 )

		# We need to store the edge groups as name : set pairs.
		edge_groups = {}

		# Now we go through the edges again in order to both fill in the
		# transition probability matrix, and also to store the indices sorted
		# by the end-node.
		for a, b, data in self.graph.edges_iter(data=True):
			# Put the edge in the dict. Its weight is log-probability
			start = self.in_edge_count[ indices[b] ]

			# Start at the beginning of the section marked off for node b.
			# If another node is already there, keep walking down the list
			# until you find a -1 meaning a node hasn't been put there yet.
			while self.in_transitions[ start ] != -1:
				if start == self.in_edge_count[ indices[b]+1 ]:
					break
				start += 1

			self.in_transition_log_probabilities[ start ] = data['weight']
			self.in_transition_pseudocounts[ start ] = data['pseudocount']

			# Store transition info in an array where the in_edge_count shows
			# the mapping stuff.
			self.in_transitions[ start ] = indices[a]

			# Now do the same for out edges
			start = self.out_edge_count[ indices[a] ]

			while self.out_transitions[ start ] != -1:
				if start == self.out_edge_count[ indices[a]+1 ]:
					break
				start += 1

			self.out_transition_log_probabilities[ start ] = data['weight']
			self.out_transition_pseudocounts[ start ] = data['pseudocount']
			self.out_transitions[ start ] = indices[b]  

			# If this edge belongs to a group, we need to add it to the
			# dictionary. We only care about forward representations of
			# the edges. 
			group = data['group']
			if group != None:
				if group in edge_groups:
					edge_groups[ group ].append( ( indices[a], indices[b] ) )
				else:
					edge_groups[ group ] = [ ( indices[a], indices[b] ) ]

		# We will organize the tied edges using three arrays. The first will be
		# the cumulative number of members in each group, to slice the later
		# arrays in the same manner as the transition arrays. The second will
		# be the index of the state the edge starts in. The third will be the
		# index of the state the edge ends in. This way, iterating across the
		# second and third lists in the slices indicated by the first list will
		# give all the edges in a group.
		total_grouped_edges = sum( map( len, edge_groups.values() ) )

		self.tied_edge_group_size = numpy.zeros( len( edge_groups.keys() )+1,
			dtype=numpy.int32 )
		self.tied_edges_starts = numpy.zeros( total_grouped_edges,
			dtype=numpy.int32 )
		self.tied_edges_ends = numpy.zeros( total_grouped_edges,
			dtype=numpy.int32 )

		# Iterate across all the grouped edges and bin them appropriately.
		for i, (name, edges) in enumerate( edge_groups.items() ):
			# Store the cumulative number of edges so far, which requires
			# adding the current number of edges (m) to the previous
			# number of edges (n)
			n = self.tied_edge_group_size[i]
			self.tied_edge_group_size[i+1] = n + len(edges)

			for j, (start, end) in enumerate( edges ):
				self.tied_edges_starts[n+j] = start
				self.tied_edges_ends[n+j] = end

		# This holds the index of the start state
		try:
			self.start_index = indices[self.start]
		except KeyError:
			raise SyntaxError( "Model.start has been deleted, leaving the \
				model with no start. Please ensure it has a start." )
		# And the end state
		try:
			self.end_index = indices[self.end]
		except KeyError:
			raise SyntaxError( "Model.end has been deleted, leaving the \
				model with no end. Please ensure it has an end." )

	def sample( self, length=0, path=False ):
		"""
		Generate a sequence from the model. Returns the sequence generated, as a
		list of emitted items. The model must have been baked first in order to 
		run this method.

		If a length is specified and the HMM is infinite (no edges to the
		end state), then that number of samples will be randomly generated.
		If the length is specified and the HMM is finite, the method will
		attempt to generate a prefix of that length. Currently it will force
		itself to not take an end transition unless that is the only path,
		making it not a true random sample on a finite model.

		WARNING: If the HMM is infinite, must specify a length to use.

		If path is True, will return a tuple of ( sample, path ), where path is
		the path of hidden states that the sample took. Otherwise, the method
		will just return the path. Note that the path length may not be the same
		length as the samples, as it will return silent states it visited, but
		they will not generate an emission.
		"""
		
		return self._sample( length, path )

	cdef list _sample( self, int length, int path ):
		"""
		Perform a run of sampling.
		"""

		cdef int i, j, k, l, li, m=len(self.states)
		cdef double cumulative_probability
		cdef double [:,:] transition_probabilities = numpy.zeros( (m,m) )
		cdef double [:] cum_probabilities = numpy.zeros( 
			len(self.out_transitions) )

		cdef int [:] out_edges = self.out_edge_count

		for k in xrange( m ):
			cumulative_probability = 0.
			for l in xrange( out_edges[k], out_edges[k+1] ):
				cumulative_probability += cexp( 
					self.out_transition_log_probabilities[l] )
				cum_probabilities[l] = cumulative_probability 

		# This holds the numerical index of the state we are currently in.
		# Start in the start state
		i = self.start_index
		
		# Record the number of samples
		cdef int n = 0
		# Define the list of emissions, and the path of hidden states taken
		cdef list emissions = [], sequence_path = []
		cdef State state
		cdef double sample

		while i != self.end_index:
			# Get the object associated with this state
			state = self.states[i]

			# Add the state to the growing path
			sequence_path.append( state )
			
			if not state.is_silent():
				# There's an emission distribution, so sample from it
				emissions.append( state.distribution.sample() )
				n += 1

			# If we've reached the specified length, return the appropriate
			# values
			if length != 0 and n >= length:
				if path:
					return [emissions, sequence_path]
				return emissions

			# What should we pick as our next state?
			# Generate a number between 0 and 1 to make a weighted decision
			# as to which state to jump to next.
			sample = random.random()
			
			# Save the last state id we were in
			j = i

			# Find out which state we're supposed to go to by comparing the
			# random number to the list of cumulative probabilities for that
			# state, and then picking the selected state.
			for k in xrange( out_edges[i], out_edges[i+1] ):
				if cum_probabilities[k] > sample:
					i = self.out_transitions[k]
					break

			# If the user specified a length, and we're not at that length, and
			# we're in an infinite HMM, we want to avoid going to the end state
			# if possible. If there is only a single probability 1 end to the
			# end state we can't avoid it, otherwise go somewhere else.
			if length != 0 and self.finite == 1 and i == self.end_index:
				# If there is only one transition...
				if len( xrange( out_edges[j], out_edges[j+1] ) ) == 1:
					# ...and that transition goes to the end of the model...
					if self.out_transitions[ out_edges[j] ] == self.end_index:
						# ... then end the sampling, as nowhere else to go.
						break

				# Take the cumulative probability of not going to the end state
				cumulative_probability = 0.
				for k in xrange( out_edges[k], out_edges[k+1] ):
					if self.out_transitions[k] != self.end_index:
						cumulative_probability += cum_probabilities[k]

				# Randomly select a number in that probability range
				sample = random.uniform( 0, cumulative_probability )

				# Select the state is corresponds to
				for k in xrange( out_edges[i], out_edges[i+1] ):
					if cum_probabilities[k] > sample:
						i = self.out_transitions[k]
						break
		
		# Done! Return either emissions, or emissions and path.
		if path:
			sequence_path.append( self.end )
			return [emissions, sequence_path]
		return emissions

	def forward( self, sequence ):
		'''
		Python wrapper for the forward algorithm, calculating probability by
		going forward through a sequence. Returns the full forward DP matrix.
		Each index i, j corresponds to the sum-of-all-paths log probability
		of starting at the beginning of the sequence, and aligning observations
		to hidden states in such a manner that observation i was aligned to
		hidden state j. Uses row normalization to dynamically scale each row
		to prevent underflow errors.

		If the sequence is impossible, will return a matrix of nans.

		input
			sequence: a list (or numpy array) of observations

		output
			A n-by-m matrix of floats, where n = len( sequence ) and
			m = len( self.states ). This is the DP matrix for the
			forward algorithm.

		See also: 
			- Silent state handling taken from p. 71 of "Biological
		Sequence Analysis" by Durbin et al., and works for anything which
		does not have loops of silent states.
			- Row normalization technique explained by 
		http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf on p. 14.
		'''

		return numpy.array( self._forward( numpy.array( sequence ) ) )

	cdef double [:,:] _forward( self, numpy.ndarray sequence ):
		"""
		Run the forward algorithm, and return the matrix of log probabilities
		of each segment being in each hidden state. 
		
		Initializes self.f, the forward algorithm DP table.
		"""

		cdef unsigned int D_SIZE = sizeof( double )
		cdef int i = 0, k, ki, l, n = len( sequence ), m = len( self.states ), j = 0
		cdef double [:,:] f, e
		cdef double log_probability
		cdef State s
		cdef Distribution d
		cdef int [:] in_edges = self.in_edge_count
		cdef double [:] c

		# Initialize the DP table. Each entry i, k holds the log probability of
		# emitting i symbols and ending in state k, starting from the start
		# state.
		f = cvarray( shape=(n+1, m), itemsize=D_SIZE, format='d' )
		c = numpy.zeros( (n+1) )

		# Initialize the emission table, which contains the probability of
		# each entry i, k holds the probability of symbol i being emitted
		# by state k 
		e = cvarray( shape=(n,self.silent_start), itemsize=D_SIZE, format='d') 
		for k in xrange( n ):
			for i in xrange( self.silent_start ):
				e[k, i] = self.states[i].distribution.log_probability(
					sequence[k] ) + self.state_weights[i]

		# We must start in the start state, having emitted 0 symbols        
		for i in xrange(m):
			f[0, i] = NEGINF
		f[0, self.start_index] = 0.

		for l in xrange( self.silent_start, m ):
			# Handle transitions between silent states before the first symbol
			# is emitted. No non-silent states have non-zero probability yet, so
			# we can ignore them.
			if l == self.start_index:
				# Start state log-probability is already right. Don't touch it.
				continue

			# This holds the log total transition probability in from 
			# all current-step silent states that can have transitions into 
			# this state.  
			log_probability = NEGINF
			for k in xrange( in_edges[l], in_edges[l+1] ):
				ki = self.in_transitions[k]
				if ki < self.silent_start or ki >= l:
					continue

				# For each current-step preceeding silent state k
				#log_probability = pair_lse( log_probability, 
				#	f[0, k] + self.transition_log_probabilities[k, l] )
				log_probability = pair_lse( log_probability,
					f[0, ki] + self.in_transition_log_probabilities[k] )

			# Update the table entry
			f[0, l] = log_probability

		for i in xrange( n ):
			for l in xrange( self.silent_start ):
				# Do the recurrence for non-silent states l
				# This holds the log total transition probability in from 
				# all previous states

				log_probability = NEGINF
				for k in xrange( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]

					# For each previous state k
					log_probability = pair_lse( log_probability,
						f[i, ki] + self.in_transition_log_probabilities[k] )

				# Now set the table entry for log probability of emitting 
				# index+1 characters and ending in state l
				f[i+1, l] = log_probability + e[i, l]

			for l in xrange( self.silent_start, m ):
				# Now do the first pass over the silent states
				# This holds the log total transition probability in from 
				# all current-step non-silent states
				log_probability = NEGINF
				for k in xrange( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]
					if ki >= self.silent_start:
						continue

					# For each current-step non-silent state k
					log_probability = pair_lse( log_probability,
						f[i+1, ki] + self.in_transition_log_probabilities[k] )

				# Set the table entry to the partial result.
				f[i+1, l] = log_probability

			for l in xrange( self.silent_start, m ):
				# Now the second pass through silent states, where we account
				# for transitions between silent states.

				# This holds the log total transition probability in from 
				# all current-step silent states that can have transitions into 
				# this state.
				log_probability = NEGINF
				for k in xrange( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]
					if ki < self.silent_start or ki >= l:
						continue

					# For each current-step preceeding silent state k
					log_probability = pair_lse( log_probability,
						f[i+1, ki] + self.in_transition_log_probabilities[k] )

				# Add the previous partial result and update the table entry
				f[i+1, l] = pair_lse( f[i+1, l], log_probability )

		# Now the DP table is filled in
		# Return the entire table
		return f

	def backward( self, sequence ):
		'''
		Python wrapper for the backward algorithm, calculating probability by
		going backward through a sequence. Returns the full forward DP matrix.
		Each index i, j corresponds to the sum-of-all-paths log probability
		of starting with observation i aligned to hidden state j, and aligning
		observations to reach the end. Uses row normalization to dynamically 
		scale each row to prevent underflow errors.

		If the sequence is impossible, will return a matrix of nans.

		input
			sequence: a list (or numpy array) of observations

		output
			A n-by-m matrix of floats, where n = len( sequence ) and
			m = len( self.states ). This is the DP matrix for the
			backward algorithm.

		See also: 
			- Silent state handling is "essentially the same" according to
		Durbin et al., so they don't bother to explain *how to actually do it*.
		Algorithm worked out from first principles.
			- Row normalization technique explained by 
		http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf on p. 14.
		'''

		return numpy.array( self._backward( numpy.array( sequence ) ) )

	cdef double [:,:] _backward( self, numpy.ndarray sequence ):
		"""
		Run the backward algorithm, and return the log probability of the given 
		sequence. Sequence is a container of symbols.
		
		Initializes self.b, the backward algorithm DP table.
		"""

		cdef unsigned int D_SIZE = sizeof( double )
		cdef int i = 0, ir, k, kr, l, li, n = len( sequence ), m = len( self.states )
		cdef double [:,:] b, e
		cdef double log_probability
		cdef State s
		cdef Distribution d
		cdef int [:] out_edges = self.out_edge_count
		cdef double [:] c

		# Initialize the DP table. Each entry i, k holds the log probability of
		# emitting the remaining len(sequence) - i symbols and ending in the end
		# state, given that we are in state k.
		b = cvarray( shape=(n+1, m), itemsize=D_SIZE, format='d' )
		c = numpy.zeros( (n+1) )

		# Initialize the emission table, which contains the probability of
		# each entry i, k holds the probability of symbol i being emitted
		# by state k 
		e = cvarray( shape=(n,self.silent_start), itemsize=D_SIZE, format='d' )

		# Calculate the emission table
		for k in xrange( n ):
			for i in xrange( self.silent_start ):
				e[k, i] = self.states[i].distribution.log_probability(
					sequence[k] ) + self.state_weights[i]

		# We must end in the end state, having emitted len(sequence) symbols
		if self.finite == 1:
			for i in xrange(m):
				b[n, i] = NEGINF
			b[n, self.end_index] = 0
		else:
			for i in xrange(self.silent_start):
				b[n, i] = 0.
			for i in xrange(self.silent_start, m):
				b[n, i] = NEGINF

		for kr in xrange( m-self.silent_start ):
			if self.finite == 0:
				break
			# Cython arrays cannot go backwards, so modify the loop to account
			# for this.
			k = m - kr - 1

			# Do the silent states' dependencies on each other.
			# Doing it in reverse order ensures that anything we can 
			# possibly transition to is already done.
			
			if k == self.end_index:
				# We already set the log-probability for this, so skip it
				continue

			# This holds the log total probability that we go to
			# current-step silent states and then continue from there to
			# finish the sequence.
			log_probability = NEGINF
			for l in xrange( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				if li < k+1:
					continue

				# For each possible current-step silent state we can go to,
				# take into account just transition probability
				log_probability = pair_lse( log_probability,
					b[n,li] + self.out_transition_log_probabilities[l] )

			# Now this is the probability of reaching the end state given we are
			# in this silent state.
			b[n, k] = log_probability

		for k in xrange( self.silent_start ):
			if self.finite == 0:
				break
			# Do the non-silent states in the last step, which depend on
			# current-step silent states.
			
			# This holds the total accumulated log probability of going
			# to such states and continuing from there to the end.
			log_probability = NEGINF
			for l in xrange( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				if li < self.silent_start:
					continue

				# For each current-step silent state, add in the probability
				# of going from here to there and then continuing on to the
				# end of the sequence.
				log_probability = pair_lse( log_probability,
					b[n, li] + self.out_transition_log_probabilities[l] )

			# Now we have summed the probabilities of all the ways we can
			# get from here to the end, so we can fill in the table entry.
			b[n, k] = log_probability

		# Now that we're done with the base case, move on to the recurrence
		for ir in xrange( n ):
			#if self.finite == 0 and ir == 0:
			#	continue
			# Cython xranges cannot go backwards properly, redo to handle
			# it properly
			i = n - ir - 1
			for kr in xrange( m-self.silent_start ):
				k = m - kr - 1

				# Do the silent states' dependency on subsequent non-silent
				# states, iterating backwards to match the order we use later.
				
				# This holds the log total probability that we go to some
				# subsequent state that emits the right thing, and then continue
				# from there to finish the sequence.
				log_probability = NEGINF
				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each subsequent non-silent state l, take into account
					# transition and emission emission probability.
					log_probability = pair_lse( log_probability,
						b[i+1, li] + self.out_transition_log_probabilities[l] +
						e[i, li] )

				# We can't go from a silent state here to a silent state on the
				# next symbol, so we're done finding the probability assuming we
				# transition straight to a non-silent state.
				b[i, k] = log_probability

			for kr in xrange( m-self.silent_start ):
				k = m - kr - 1

				# Do the silent states' dependencies on each other.
				# Doing it in reverse order ensures that anything we can 
				# possibly transition to is already done.
				
				# This holds the log total probability that we go to
				# current-step silent states and then continue from there to
				# finish the sequence.
				log_probability = NEGINF
				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li < k+1:
						continue

					# For each possible current-step silent state we can go to,
					# take into account just transition probability
					log_probability = pair_lse( log_probability,
						b[i, li] + self.out_transition_log_probabilities[l] )

				# Now add this probability in with the probability accumulated
				# from transitions to subsequent non-silent states.
				b[i, k] = pair_lse( log_probability, b[i, k] )

			for k in xrange( self.silent_start ):
				# Do the non-silent states in the current step, which depend on
				# subsequent non-silent states and current-step silent states.
				
				# This holds the total accumulated log probability of going
				# to such states and continuing from there to the end.
				log_probability = NEGINF
				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each subsequent non-silent state l, take into account
					# transition and emission emission probability.
					log_probability = pair_lse( log_probability,
						b[i+1, li] + self.out_transition_log_probabilities[l] +
						e[i, li] )

				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li < self.silent_start:
						continue

					# For each current-step silent state, add in the probability
					# of going from here to there and then continuing on to the
					# end of the sequence.
					log_probability = pair_lse( log_probability,
						b[i, li] + self.out_transition_log_probabilities[l] )

				# Now we have summed the probabilities of all the ways we can
				# get from here to the end, so we can fill in the table entry.
				b[i, k] = log_probability

		# Now the DP table is filled in. 
		# Return the entire table.
		return b

	def forward_backward( self, sequence, tie=False ):
		"""
		Implements the forward-backward algorithm. This is the sum-of-all-paths
		log probability that you start at the beginning of the sequence, align
		observation i to silent state j, and then continue on to the end.
		Simply, it is the probability of emitting the observation given the
		state and then transitioning one step.

		If the sequence is impossible, will return (None, None)

		input
			sequence: a list (or numpy array) of observations

		output
			A tuple of the estimated log transition probabilities, and
			the DP matrix for the FB algorithm. The DP matrix has
			n rows and m columns where n is the number of observations,
			and m is the number of non-silent states.

			* The estimated log transition probabilities are a m-by-m 
			matrix where index i, j indicates the log probability of 
			transitioning from state i to state j.

			* The DP matrix for the FB algorithm contains the sum-of-all-paths
			probability as described above.

		See also: 
			- Forward and backward algorithm implementations. A comprehensive
			description of the forward, backward, and forward-background
			algorithm is here: 
			http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
		"""

		return self._forward_backward( numpy.array( sequence ), tie )

	cdef tuple _forward_backward( self, numpy.ndarray sequence, int tie ):
		"""
		Actually perform the math here.
		"""

		cdef int i, k, j, l, ki, li
		cdef int m=len(self.states), n=len(sequence)
		cdef double [:,:] e, f, b
		cdef double [:,:] expected_transitions = numpy.zeros((m, m))
		cdef double [:,:] emission_weights = numpy.zeros((n, self.silent_start))

		cdef double log_sequence_probability, log_probability
		cdef double log_transition_emission_probability_sum
		cdef double norm

		cdef int [:] out_edges = self.out_edge_count
		cdef int [:] tied_states = self.tied_state_count

		cdef State s
		cdef Distribution d 

		transition_log_probabilities = numpy.zeros((m,m)) + NEGINF

		# Initialize the emission table, which contains the probability of
		# each entry i, k holds the probability of symbol i being emitted
		# by state k 
		e = numpy.zeros((n, self.silent_start))

		# Fill in both the F and B DP matrices.
		f = self.forward( sequence )
		b = self.backward( sequence )

		# Calculate the emission table
		for k in xrange( n ):
			for i in xrange( self.silent_start ):
				e[k, i] = self.states[i].distribution.log_probability(
					sequence[k] ) + self.state_weights[i]

		if self.finite == 1:
			log_sequence_probability = f[ n, self.end_index ]
		else:
			log_sequence_probability = NEGINF
			for i in xrange( self.silent_start ):
				log_sequence_probability = pair_lse( 
					log_sequence_probability, f[ n, i ] )
		
		# Is the sequence impossible? If so, don't bother calculating any more.
		if log_sequence_probability == NEGINF:
			print( "Warning: Sequence is impossible." )
			return ( None, None )

		for k in xrange( m ):
			# For each state we could have come from
			for l in xrange( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				if li >= self.silent_start:
					continue

				# For each state we could go to (and emit a character)
				# Sum up probabilities that we later normalize by 
				# probability of sequence.
				log_transition_emission_probability_sum = NEGINF

				for i in xrange( n ):
					# For each character in the sequence
					# Add probability that we start and get up to state k, 
					# and go k->l, and emit the symbol from l, and go from l
					# to the end.
					log_transition_emission_probability_sum = pair_lse( 
						log_transition_emission_probability_sum, 
						f[i, k] + self.out_transition_log_probabilities[l] +
						e[i, li] + b[i+1, li] )

				# Now divide by probability of the sequence to make it given
				# this sequence, and add as this sequence's contribution to 
				# the expected transitions matrix's k, l entry.
				expected_transitions[k, li] += cexp(
					log_transition_emission_probability_sum - 
					log_sequence_probability )

			for l in xrange( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				if li < self.silent_start:
					continue

				# For each silent state we can go to on the same character
				# Sum up probabilities that we later normalize by 
				# probability of sequence.

				log_transition_emission_probability_sum = NEGINF
				for i in xrange( n+1 ):
					# For each row in the forward DP table (where we can
					# have transitions to silent states) of which we have 1 
					# more than we have symbols...
						
					# Add probability that we start and get up to state k, 
					# and go k->l, and go from l to the end. In this case, 
					# we use forward and backward entries from the same DP 
					# table row, since no character is being emitted.
					log_transition_emission_probability_sum = pair_lse( 
						log_transition_emission_probability_sum, 
						f[i, k] + self.out_transition_log_probabilities[l]
						+ b[i, li] )
					
				# Now divide by probability of the sequence to make it given
				# this sequence, and add as this sequence's contribution to 
				# the expected transitions matrix's k, l entry.
				expected_transitions[k, li] += cexp(
					log_transition_emission_probability_sum -
					log_sequence_probability )
				
			if k < self.silent_start:
				# Now think about emission probabilities from this state
						  
				for i in xrange( n ):
					# For each symbol that came out
		   
					# What's the weight of this symbol for that state?
					# Probability that we emit index characters and then 
					# transition to state l, and that from state l we  
					# continue on to emit len(sequence) - (index + 1) 
					# characters, divided by the probability of the 
					# sequence under the model.
					# According to http://www1.icsi.berkeley.edu/Speech/
					# docs/HTKBook/node7_mn.html, we really should divide by
					# sequence probability.

					emission_weights[i,k] = f[i+1, k] + b[i+1, k] - \
						log_sequence_probability
		
		cdef int [:] visited
		cdef double tied_state_log_probability
		if tie == 1:
			visited = numpy.zeros( self.silent_start, dtype=numpy.int32 )

			for k in xrange( self.silent_start ):
				# Check to see if we have visited this a state within the set of
				# tied states this state belongs yet. If not, this is the first
				# state and we can calculate the tied probabilities here.
				if visited[k] == 1:
					continue
				visited[k] = 1

				# Set that we have visited all of the other members of this set
				# of tied states.
				for l in xrange( tied_states[k], tied_states[k+1] ):
					li = self.tied[l]
					visited[li] = 1

				for i in xrange( n ):
					# Begin the probability sum with the log probability of 
					# being in the current state.
					tied_state_log_probability = emission_weights[i, k]

					# Go through all the states this state is tied with, and
					# add up the probability of being in any of them, and
					# updated the visited list.
					for l in xrange( tied_states[k], tied_states[k+1] ):
						li = self.tied[l]
						tied_state_log_probability = pair_lse( 
							tied_state_log_probability, emission_weights[i, li] )

					# Now update them with the retrieved value
					for l in xrange( tied_states[k], tied_states[k+1] ):
						li = self.tied[l]
						emission_weights[i, li] = tied_state_log_probability

					# Update the initial state we started with
					emission_weights[i, k] = tied_state_log_probability

		return numpy.array( expected_transitions ), \
			numpy.array( emission_weights )

	def log_probability( self, sequence, path=None ):
		'''
		Calculate the log probability of a single sequence. If a path is
		provided, calculate the log probability of that sequence given
		the path.
		'''

		if path:
			return self._log_probability_of_path( numpy.array( sequence ),
				numpy.array( path ) )
		return self._log_probability( numpy.array( sequence ) )

	cdef double _log_probability( self, numpy.ndarray sequence ):
		'''
		Calculate the probability here, in a cython optimized function.
		'''

		cdef int i
		cdef double log_probability_sum
		cdef double [:,:] f 

		f = self.forward( sequence )
		if self.finite == 1:
			log_probability_sum = f[ len(sequence), self.end_index ]
		else:
			log_probability_sum = NEGINF
			for i in xrange( self.silent_start ):
				log_probability_sum = pair_lse( 
					log_probability_sum, f[ len(sequence), i ] )

		return log_probability_sum

	cdef double _log_probability_of_path( self, numpy.ndarray sequence,
		State [:] path ):
		'''
		Calculate the probability of a sequence, given the path it took through
		the model.
		'''

		cdef int i=0, idx, j, ji, l, li, ki, m=len(self.states)
		cdef int p=len(path), n=len(sequence)
		cdef dict indices = { self.states[i]: i for i in xrange( m ) }
		cdef State state

		cdef int [:] out_edges = self.out_edge_count

		cdef double log_score = 0

		# Iterate over the states in the path, as the path needs to be either
		# equal in length or longer than the sequence, depending on if there
		# are silent states or not.
		for j in xrange( 1, p ):
			# Add the transition probability first, because both silent and
			# character generating states have to do the transition. So find
			# the index of the last state, and see if there are any out
			# edges from that state to the current state. This operation
			# requires time proportional to the number of edges leaving the
			# state, due to the way the sparse representation is set up.
			ki = indices[ path[j-1] ]
			ji = indices[ path[j] ]

			for l in xrange( out_edges[ki], out_edges[ki+1] ):
				li = self.out_transitions[l]
				if li == ji:
					log_score += self.out_transition_log_probabilities[l]
					break
				if l == out_edges[ki+1]-1:
					return NEGINF

			# If the state is not silent, then add the log probability of
			# emitting that observation from this state.
			if not path[j].is_silent():
				log_score += path[j].distribution.log_probability( 
					sequence[i] )
				i += 1

		return log_score

	def viterbi( self, sequence ):
		'''
		Run the Viterbi algorithm on the sequence given the model. This finds
		the ML path of hidden states given the sequence. Returns a tuple of the
		log probability of the ML path, or (-inf, None) if the sequence is
		impossible under the model. If a path is returned, it is a list of
		tuples of the form (sequence index, state object).

		This is fundamentally the same as the forward algorithm using max
		instead of sum, except the traceback is more complicated, because
		silent states in the current step can trace back to other silent states
		in the current step as well as states in the previous step.

		input
			sequence: a list (or numpy array) of observations

		output
			A tuple of the log probabiliy of the ML path, and the sequence of
			hidden states that comprise the ML path.

		See also: 
			- Viterbi implementation described well in the wikipedia article
			http://en.wikipedia.org/wiki/Viterbi_algorithm
		'''

		return self._viterbi( numpy.array( sequence ) )

	cdef tuple _viterbi(self, numpy.ndarray sequence):
		"""		
		This fills in self.v, the Viterbi algorithm DP table.
		
		This is fundamentally the same as the forward algorithm using max
		instead of sum, except the traceback is more complicated, because silent
		states in the current step can trace back to other silent states in the
		current step as well as states in the previous step.
		"""
		cdef unsigned int I_SIZE = sizeof( int ), D_SIZE = sizeof( double )

		cdef unsigned int n = sequence.shape[0], m = len(self.states)
		cdef double p
		cdef int i, l, k, ki
		cdef int [:,:] tracebackx, tracebacky
		cdef double [:,:] v, e
		cdef double state_log_probability
		cdef Distribution d
		cdef State s
		cdef int[:] in_edges = self.in_edge_count

		# Initialize the DP table. Each entry i, k holds the log probability of
		# emitting i symbols and ending in state k, starting from the start
		# state, along the most likely path.
		v = cvarray( shape=(n+1,m), itemsize=D_SIZE, format='d' )

		# Initialize the emission table, which contains the probability of
		# each entry i, k holds the probability of symbol i being emitted
		# by state k 
		e = cvarray( shape=(n,self.silent_start), itemsize=D_SIZE, format='d' )

		# Initialize two traceback matricies. Each entry in tracebackx points
		# to the x index on the v matrix of the next entry. Same for the
		# tracebacky matrix.
		tracebackx = cvarray( shape=(n+1,m), itemsize=I_SIZE, format='i' )
		tracebacky = cvarray( shape=(n+1,m), itemsize=I_SIZE, format='i' )

		for k in xrange( n ):
			for i in xrange( self.silent_start ):
				e[k, i] = self.states[i].distribution.log_probability( 
					sequence[k] ) + self.state_weights[i]

		# We catch when we trace back to (0, self.start_index), so we don't need
		# a traceback there.
		for i in xrange( m ):
			v[0, i] = NEGINF
		v[0, self.start_index] = 0
		# We must start in the start state, having emitted 0 symbols

		for l in xrange( self.silent_start, m ):
			# Handle transitions between silent states before the first symbol
			# is emitted. No non-silent states have non-zero probability yet, so
			# we can ignore them.
			if l == self.start_index:
				# Start state log-probability is already right. Don't touch it.
				continue

			for k in xrange( in_edges[l], in_edges[l+1] ):
				ki = self.in_transitions[k]
				if ki < self.silent_start or ki >= l:
					continue

				# For each current-step preceeding silent state k
				# This holds the log-probability coming that way
				state_log_probability = v[0, ki] + \
					self.in_transition_log_probabilities[k]

				if state_log_probability > v[0, l]:
					# New winner!
					v[0, l] = state_log_probability
					tracebackx[0, l] = 0
					tracebacky[0, l] = ki

		for i in xrange( n ):
			for l in xrange( self.silent_start ):
				# Do the recurrence for non-silent states l
				# Start out saying the best likelihood we have is -inf
				v[i+1, l] = NEGINF
				
				for k in xrange( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]

					# For each previous state k
					# This holds the log-probability coming that way
					state_log_probability = v[i, ki] + \
						self.in_transition_log_probabilities[k] + e[i, l]

					if state_log_probability > v[i+1, l]:
						# Best to come from there to here
						v[i+1, l] = state_log_probability
						tracebackx[i+1, l] = i
						tracebacky[i+1, l] = ki

			for l in xrange( self.silent_start, m ):
				# Now do the first pass over the silent states, finding the best
				# current-step non-silent state they could come from.
				# Start out saying the best likelihood we have is -inf
				v[i+1, l] = NEGINF

				for k in xrange( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]
					if ki >= self.silent_start:
						continue

					# For each current-step non-silent state k
					# This holds the log-probability coming that way
					state_log_probability = v[i+1, ki] + \
						self.in_transition_log_probabilities[k]

					if state_log_probability > v[i+1, l]:
						# Best to come from there to here
						v[i+1, l] = state_log_probability
						tracebackx[i+1, l] = i+1
						tracebacky[i+1, l] = ki

			for l in xrange( self.silent_start, m ):
				# Now the second pass through silent states, where we check the
				# silent states that could potentially reach here and see if
				# they're better than the non-silent states we found.

				for k in xrange( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]
					if ki < self.silent_start or ki >= l:
						continue

					# For each current-step preceeding silent state k
					# This holds the log-probability coming that way
					state_log_probability = v[i+1, ki] + \
						self.in_transition_log_probabilities[k]

					if state_log_probability > v[i+1, l]:
						# Best to come from there to here
						v[i+1, l] = state_log_probability
						tracebackx[i+1, l] = i+1
						tracebacky[i+1, l] = ki

		# Now the DP table is filled in. If this is a finite model, get the
		# log likelihood of ending up in the end state after following the
		# ML path through the model. If an infinite sequence, find the state
		# which the ML path ends in, and begin there.
		cdef int end_index
		cdef double log_likelihood

		if self.finite == 1:
			log_likelihood = v[n, self.end_index]
			end_index = self.end_index
		else:
			end_index = numpy.argmax( v[n] )
			log_likelihood = v[n, end_index ]

		if log_likelihood == NEGINF:
			# The path is impossible, so don't even try a traceback. 
			return ( log_likelihood, None )

		# Otherwise, do the traceback
		# This holds the path, which we construct in reverse order
		cdef list path = []
		cdef int px = n, py = end_index, npx

		# This holds our current position (character, state) AKA (i, k).
		# We start at the end state
		while px != 0 or py != self.start_index:
			# Until we've traced back to the start...
			# Put the position in the path, making sure to look up the state
			# object to use instead of the state index.
			path.append( ( py, self.states[py] ) )

			# Go backwards
			npx = tracebackx[px, py]
			py = tracebacky[px, py]
			px = npx

		# We've now reached the start (if we didn't raise an exception because
		# we messed up the traceback)
		# Record that we start at the start
		path.append( (py, self.states[py] ) )

		# Flip the path the right way around
		path.reverse()

		# Return the log-likelihood and the right-way-arounded path
		return ( log_likelihood, path )

	def maximum_a_posteriori( self, sequence ):
		"""
		MAP decoding is an alternative to viterbi decoding, which returns the
		most likely state for each observation, based on the forward-backward
		algorithm. This is also called posterior decoding. This method is
		described on p. 14 of http://ai.stanford.edu/~serafim/CS262_2007/
		notes/lecture5.pdf

		WARNING: This may produce impossible sequences.
		"""

		return self._maximum_a_posteriori( numpy.array( sequence ) )

	
	cdef tuple _maximum_a_posteriori( self, numpy.ndarray sequence ):
		"""
		Actually perform the math here. Instead of calling forward-backward
		to get the emission weights, it's calculated here so that time isn't
		wasted calculating the transition counts. 
		"""

		cdef int i, k, l, li
		cdef int m=len(self.states), n=len(sequence)
		cdef double [:,:] f, b
		cdef double [:,:] emission_weights = numpy.zeros((n, self.silent_start))
		cdef int [:] tied_states = self.tied_state_count

		cdef double log_sequence_probability


		# Fill in both the F and B DP matrices.
		f = self.forward( sequence )
		b = self.backward( sequence )

		# Find out the probability of the sequence
		if self.finite == 1:
			log_sequence_probability = f[ n, self.end_index ]
		else:
			log_sequence_probability = NEGINF
			for i in xrange( self.silent_start ):
				log_sequence_probability = pair_lse( 
					log_sequence_probability, f[ n, i ] )
		
		# Is the sequence impossible? If so, don't bother calculating any more.
		if log_sequence_probability == NEGINF:
			print( "Warning: Sequence is impossible." )
			return ( None, None )

		for k in xrange( m ):				
			if k < self.silent_start:				  
				for i in xrange( n ):
					# For each symbol that came out
					# What's the weight of this symbol for that state?
					# Probability that we emit index characters and then 
					# transition to state l, and that from state l we  
					# continue on to emit len(sequence) - (index + 1) 
					# characters, divided by the probability of the 
					# sequence under the model.
					# According to http://www1.icsi.berkeley.edu/Speech/
					# docs/HTKBook/node7_mn.html, we really should divide by
					# sequence probability.
					emission_weights[i,k] = f[i+1, k] + b[i+1, k] - \
						log_sequence_probability

		cdef list path = [ ( self.start_index, self.start ) ]
		cdef double maximum_emission_weight
		cdef double log_probability_sum = 0
		cdef int maximum_index

		# Go through each symbol and determine what the most likely state
		# that it came from is.
		for k in xrange( n ):
			maximum_index = -1
			maximum_emission_weight = NEGINF

			# Go through each hidden state and see which one has the maximal
			# weight for emissions. Tied states are not taken into account
			# here, because we are not performing training.
			for l in xrange( self.silent_start ):
				if emission_weights[k, l] > maximum_emission_weight:
					maximum_emission_weight = emission_weights[k, l]
					maximum_index = l

			path.append( ( maximum_index, self.states[maximum_index] ) )
			log_probability_sum += maximum_emission_weight 

		path.append( ( self.end_index, self.end ) )

		return log_probability_sum, path

	def write(self, stream):
		"""
		Write out the HMM to the given stream in a format more sane than pickle.
		
		HMM must have been baked.
		
		HMM is written as  "<identity> <name> <weight> <Distribution>" tuples 
		which can be directly evaluated by the eval method. This makes them 
		both human readable, and keeps the code for it super simple.
		
		The start state is the one named "<hmm name>-start" and the end state is
		the one named "<hmm name>-end". Start and end states are always silent.
		
		Having the number of states on the first line makes the format harder 
		for humans to write, but saves us from having to write a real 
		backtracking parser.
		"""
		
		print("Warning: Writing currently only writes out the model structure,\
			and not any information about tied edges or distributions.")
		
		# Change our name to remove all whitespace, as this causes issues
		# with the parsing later on.
		self.name = self.name.replace( " ", "_" )

		# Write our name.
		stream.write("{} {}\n".format(self.name, len(self.states)))
		
		for state in sorted(self.states, key=lambda s: s.name):
			# Write each state in order by name
			state.write(stream)
			
		# Get transitions.
		# Each is a tuple (from index, to index, log probability, pseudocount)
		transitions = []
		
		for k in xrange( len(self.states) ):
			for l in xrange( self.out_edge_count[k], self.out_edge_count[k+1] ):
				li = self.out_transitions[l]
				log_probability = self.out_transition_log_probabilities[l]
				pseudocount = self.out_transition_pseudocounts[l]

				transitions.append( (k, li, log_probability, pseudocount) )
			
		for (from_index, to_index, log_probability, pseudocount) in transitions:
			
			# Write each transition, using state names instead of indices.
			# This requires lookups and makes state names need to be unique, but
			# it's more human-readable and human-writeable.
			
			# Get the name of the state we're leaving
			from_name = self.states[from_index].name.replace( " ", "_" )
			from_id = self.states[from_index].identity
			
			# And the one we're going to
			to_name = self.states[to_index].name.replace( " ", "_" )
			to_id = self.states[to_index].identity

			# And the probability
			probability = exp(log_probability)
			
			# Write it out
			stream.write("{} {} {} {} {} {}\n".format(
				from_name, to_name, probability, pseudocount, from_id, to_id))
			
	@classmethod
	def read(cls, stream, verbose=False):
		"""
		Read a HMM from the given stream, in the format used by write(). The 
		stream must end at the end of the data defining the HMM.
		"""
		
		# Read the name and state count (first line)
		header = stream.readline()
		
		if header == "":
			raise EOFError("EOF reading HMM header")
		
		# Spilt out the parts of the headr
		parts = header.strip().split()
		
		# Get the HMM name
		name = parts[0]
		
		# Get the number of states to read
		num_states = int(parts[-1])
		
		# Read and make the states.
		# Keep a dict of states by id
		states = {}
		
		for i in xrange(num_states):
			# Read in a state
			state = State.read(stream)
			
			# Store it in the state dict
			states[state.identity] = state

			# We need to find the start and end states before we can make the HMM.
			# Luckily, we know their names.
			if state.name == "{}-start".format( name ):
				start_state = state
			if state.name == "{}-end".format( name ):
				end_state = state
			
		# Make the HMM object to populate
		hmm = cls(name=name, start=start_state, end=end_state)
		
		for state in states.itervalues():
			if state != start_state and state != end_state:
				# This state isn't already in the HMM, so add it.
				hmm.add_state(state)

		# Now do the transitions (all the rest of the lines)
		for line in stream:
			# Pull out the from state name, to state name, and probability 
			# string
			( from_name, to_name, probability_string, pseudocount_string,
				from_id, to_id ) = line.strip().split()
			
			# Make the probability as a float
			probability = float(probability_string)
			
			# Make the pseudocount a float too
			pseudocount = float(pseudocount_string)

			# Look up the states and add the transition
			hmm.add_transition(
				states[from_id], states[to_id], probability, pseudocount )

		# Now our HMM is done.
		# Bake and return it.
		hmm.bake( merge=None )
		return hmm
	
	@classmethod
	def from_matrix( cls, transition_probabilities, distributions, starts, ends,
		state_names=None, name=None ):
		"""
		Take in a 2D matrix of floats of size n by n, which are the transition
		probabilities to go from any state to any other state. May also take in
		a list of length n representing the names of these nodes, and a model
		name. Must provide the matrix, and a list of size n representing the
		distribution you wish to use for that state, a list of size n indicating
		the probability of starting in a state, and a list of size n indicating
		the probability of ending in a state.

		For example, if you wanted a model with two states, A and B, and a 0.5
		probability of switching to the other state, 0.4 probability of staying
		in the same state, and 0.1 probability of ending, you'd write the HMM
		like this:

		matrix = [ [ 0.4, 0.5 ], [ 0.4, 0.5 ] ]
		distributions = [NormalDistribution(1, .5), NormalDistribution(5, 2)]
		starts = [ 1., 0. ]
		ends = [ .1., .1 ]
		state_names= [ "A", "B" ]

		model = Model.from_matrix( matrix, distributions, starts, ends, 
			state_names, name="test_model" )
		"""

		# Build the initial model
		model = Model( name=name )

		# Build state objects for every state with the appropriate distribution
		states = [ State( distribution, name=name ) for name, distribution in
			izip( state_names, distributions) ]

		n = len( states )

		# Add all the states to the model
		for state in states:
			model.add_state( state )

		# Connect the start of the model to the appropriate state
		for i, prob in enumerate( starts ):
			if prob != 0:
				model.add_transition( model.start, states[i], prob )

		# Connect all states to each other if they have a non-zero probability
		for i in xrange( n ):
			for j, prob in enumerate( transition_probabilities[i] ):
				if prob != 0.:
					model.add_transition( states[i], states[j], prob )

		# Connect states to the end of the model if a non-zero probability 
		for i, prob in enumerate( ends ):
			if prob != 0:
				model.add_transition( states[j], model.end, prob )

		model.bake()
		return model

	def train( self, sequences, stop_threshold=1E-9, min_iterations=0,
		max_iterations=None, algorithm='baum-welch', verbose=True,
		transition_pseudocount=0, use_pseudocount=False, edge_inertia=0.0,
		distribution_inertia=0.0 ):
		"""
		Given a list of sequences, performs re-estimation on the model
		parameters. The two supported algorithms are "baum-welch" and
		"viterbi," indicating their respective algorithm. 

		Use either a uniform transition_pseudocount, or the
		previously specified ones by toggling use_pseudocount if pseudocounts
		are needed. edge_inertia can make the new edge parameters be a mix of
		new parameters and the old ones, and distribution_inertia does the same
		thing for distributions instead of transitions.

		Baum-Welch: Iterates until the log of the "score" (total likelihood of 
		all sequences) changes by less than stop_threshold. Returns the final 
		log score.
	
		Always trains for at least min_iterations, and terminate either when
		reaching max_iterations, or the training improvement is smaller than
		stop_threshold.

		Viterbi: Training performed by running each sequence through the
		viterbi decoding algorithm. Edge weight re-estimation is done by 
		recording the number of times a hidden state transitions to another 
		hidden state, and using the percentage of time that edge was taken.
		Emission re-estimation is done by retraining the distribution on
		every sample tagged as belonging to that state.

		Baum-Welch training is usually the more accurate method, but takes
		significantly longer. Viterbi is a good for situations in which
		accuracy can be sacrificed for time.
		"""

		# Convert the boolean into an integer for downstream use.
		use_pseudocount = int( use_pseudocount )

		if algorithm.lower() == 'labelled' or algorithm.lower() == 'labeled':
			for i, sequence in enumerate(sequences):
				sequences[i] = ( numpy.array( sequence[0] ), sequence[1] )

			# If calling the labelled training algorithm, then sequences is a
			# list of tuples of sequence, path pairs, not a list of sequences.
			# The log probability sum is the log-sum-exp of the log
			# probabilities of all members of the sequence. In this case,
			# sequences is made up of ( sequence, path ) tuples, instead of
			# just sequences.
			log_probability_sum = log_probability( self, sequences )
		
			self._train_labelled( sequences, transition_pseudocount, 
				use_pseudocount, edge_inertia, distribution_inertia )
		else:
			# Take the logsumexp of the log probabilities of the sequences.
			# Since sequences is just a list of sequences now, we can map
			# the log probability function directly onto it.
			log_probability_sum = log_probability( self, sequences )

		# Cast everything as a numpy array for input into the other possible
		# training algorithms.
		sequences = numpy.array( sequences )
		for i, sequence in enumerate( sequences ):
			sequences[i] = numpy.array( sequence )

		if algorithm.lower() == 'viterbi':
			self._train_viterbi( sequences, transition_pseudocount,
				use_pseudocount, edge_inertia, distribution_inertia )

		elif algorithm.lower() == 'baum-welch':
			self._train_baum_welch( sequences, stop_threshold,
				min_iterations, max_iterations, verbose, 
				transition_pseudocount, use_pseudocount, edge_inertia,
				distribution_inertia )

		# If using the labeled training algorithm, then calculate the new
		# probability sum across the path it chose, instead of the
		# sum-of-all-paths probability.
		if algorithm.lower() == 'labelled' or algorithm.lower() == 'labeled':
			# Since there are labels for this training, make sure to calculate
			# the log probability given the path. 
			trained_log_probability_sum = log_probability( self, sequences )
		else:
			# Given that there are no labels, calculate the logsumexp by
			# mapping the log probability function directly onto the sequences.
			trained_log_probability_sum = log_probability( self, sequences )

		# Calculate the difference between the two measurements.
		improvement = trained_log_probability_sum - log_probability_sum

		if verbose:
			print "Total Training Improvement: ", improvement
		return improvement

	def _train_baum_welch(self, sequences, stop_threshold, min_iterations, 
		max_iterations, verbose, transition_pseudocount, use_pseudocount,
		edge_inertia, distribution_inertia ):
		"""
		Given a list of sequences, perform Baum-Welch iterative re-estimation on
		the model parameters.
		
		Iterates until the log of the "score" (total likelihood of all 
		sequences) changes by less than stop_threshold. Returns the final log
		score.
		
		Always trains for at least min_iterations.
		"""

		# How many iterations of training have we done (counting the first)
		iteration, improvement = 0, float("+inf")
		last_log_probability_sum = log_probability( self, sequences )

		while improvement > stop_threshold or iteration < min_iterations:
			if max_iterations and iteration >= max_iterations:
				break 

			# Perform an iteration of Baum-Welch training.
			self._train_once_baum_welch( sequences, transition_pseudocount, 
				use_pseudocount, edge_inertia, distribution_inertia )

			# Increase the iteration counter by one.
			iteration += 1

			# Calculate the improvement yielded by that iteration of
			# Baum-Welch. First, we must calculate probability of sequences
			# after training, which is just the logsumexp of the log
			# probabilities of the sequence.
			trained_log_probability_sum = log_probability( self, sequences )

			# The improvement is the difference between the log probability of
			# all the sequences after training, and the log probability before
			# training.
			improvement = trained_log_probability_sum - last_log_probability_sum
			last_log_probability_sum = trained_log_probability_sum

			if verbose:
				print( "Training improvement: {}".format(improvement) )
			
	cdef void _train_once_baum_welch(self, numpy.ndarray sequences, 
		double transition_pseudocount, int use_pseudocount, 
		double edge_inertia, double distribution_inertia ):
		"""
		Implements one iteration of the Baum-Welch algorithm, as described in:
		http://www.cs.cmu.edu/~durand/03-711/2006/Lectures/hmm-bw.pdf
			
		Returns the log of the "score" under the *previous* set of parameters. 
		The score is the sum of the likelihoods of all the sequences.
		"""        

		cdef double [:,:] transition_log_probabilities 
		cdef double [:,:] expected_transitions, e, f, b
		cdef double [:,:] emission_weights
		cdef numpy.ndarray sequence
		cdef double log_sequence_probability
		cdef double sequence_probability_sum
		cdef int k, i, l, li, m = len( self.states ), n, observation=0
		cdef int characters_so_far = 0
		cdef object symbol
		cdef double [:] weights

		cdef int [:] out_edges = self.out_edge_count
		cdef int [:] in_edges = self.in_edge_count

		# Define several helped variables.
		cdef int [:] visited
		cdef int [:] tied_states = self.tied_state_count

		# Find the expected number of transitions between each pair of states, 
		# given our data and our current parameters, but allowing the paths 
		# taken to vary. (Indexed: from, to)
		expected_transitions = numpy.zeros(( m, m ))

		for sequence in sequences:
			n = len( sequence )
			# Calculate the emission table
			e = numpy.zeros(( n, self.silent_start )) 
			for k in xrange( n ):
				for i in xrange( self.silent_start ):
					e[k, i] = self.states[i].distribution.log_probability( 
						sequence[k] ) + self.state_weights[i]

			# Get the overall log probability of the sequence, and fill in the
			# the forward DP matrix.
			f = self.forward( sequence )
			if self.finite == 1:
				log_sequence_probability = f[ n, self.end_index ]
			else:
				log_sequence_probability = NEGINF
				for i in xrange( self.silent_start ):
					log_sequence_probability = pair_lse( f[n, i],
						log_sequence_probability )

			# Is the sequence impossible? If so, we can't train on it, so skip 
			# it
			if log_sequence_probability == NEGINF:
				print( "Warning: skipped impossible sequence {}".format(sequence) )
				continue

			# Fill in the backward DP matrix.
			b = self.backward(sequence)

			# Set the visited array to entirely 0s, meaning we haven't visited
			# any states yet. 
			visited = numpy.zeros( self.silent_start, dtype=numpy.int32 )
			for k in xrange( m ):
				# For each state we could have come from
				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each state we could go to (and emit a character)
					# Sum up probabilities that we later normalize by 
					# probability of sequence.
					log_transition_emission_probability_sum = NEGINF
					for i in xrange( n ):
						# For each character in the sequence
						# Add probability that we start and get up to state k, 
						# and go k->l, and emit the symbol from l, and go from l
						# to the end.
						log_transition_emission_probability_sum = pair_lse( 
							log_transition_emission_probability_sum, 
							f[i, k] + 
							self.out_transition_log_probabilities[l] + 
							e[i, li] + b[ i+1, li] )

					# Now divide by probability of the sequence to make it given
					# this sequence, and add as this sequence's contribution to 
					# the expected transitions matrix's k, l entry.
					expected_transitions[k, li] += cexp(
						log_transition_emission_probability_sum - 
						log_sequence_probability)

				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li < self.silent_start:
						continue
					# For each silent state we can go to on the same character
					# Sum up probabilities that we later normalize by 
					# probability of sequence.
					log_transition_emission_probability_sum = NEGINF
					for i in xrange( n + 1 ):
						# For each row in the forward DP table (where we can
						# have transitions to silent states) of which we have 1 
						# more than we have symbols...

						# Add probability that we start and get up to state k, 
						# and go k->l, and go from l to the end. In this case, 
						# we use forward and backward entries from the same DP 
						# table row, since no character is being emitted.
						log_transition_emission_probability_sum = pair_lse( 
							log_transition_emission_probability_sum, 
							f[i, k] + self.out_transition_log_probabilities[l] 
							+ b[i, li] )

					# Now divide by probability of the sequence to make it given
					# this sequence, and add as this sequence's contribution to 
					# the expected transitions matrix's k, l entry.
					expected_transitions[k, li] += cexp(
						log_transition_emission_probability_sum -
						log_sequence_probability )

				if k < self.silent_start:
					# If another state in the set of tied states has already
					# been visited, we don't want to retrain.
					if visited[k] == 1:
						continue

					# Mark that we've visited this state
					visited[k] = 1

					# Mark that we've visited all other states in this state
					# group.
					for l in xrange( tied_states[k], tied_states[k+1] ):
						li = self.tied[l]
						visited[li] = 1

					# Now think about emission probabilities from this state
					weights = numpy.zeros( n )

					for i in xrange( n ):
						# For each symbol that came out
						# What's the weight of this symbol for that state?
						# Probability that we emit index characters and then 
						# transition to state l, and that from state l we  
						# continue on to emit len(sequence) - (index + 1) 
						# characters, divided by the probability of the 
						# sequence under the model.
						# According to http://www1.icsi.berkeley.edu/Speech/
						# docs/HTKBook/node7_mn.html, we really should divide by
						# sequence probability.
						weights[i] = cexp( f[i+1, k] + b[i+1, k] - 
							log_sequence_probability )
						
						for l in xrange( tied_states[k], tied_states[k+1] ):
							li = self.tied[l]
							weights[i] += cexp( f[i+1, li] + b[i+1, li] -
								log_sequence_probability )

					self.states[k].distribution.summarize( sequence, weights )

		# We now have expected_transitions taking into account all sequences.
		# And a list of all emissions, and a weighting of each emission for each
		# state
		# Normalize transition expectations per row (so it becomes transition 
		# probabilities)
		# See http://stackoverflow.com/a/8904762/402891
		# Only modifies transitions for states a transition was observed from.
		cdef double [:] norm = numpy.zeros( m )
		cdef double probability

		cdef int [:] tied_edges = self.tied_edge_group_size
		cdef double tied_edge_probability 
		# Go through the tied state groups and add transitions from each member
		# in the group to the other members of the group.
		# For each group defined.
		for k in xrange( len( tied_edges )-1 ):
			tied_edge_probability = 0.

			# For edge in this group, get the sum of the edges
			for l in xrange( tied_edges[k], tied_edges[k+1] ):
				start = self.tied_edges_starts[l]
				end = self.tied_edges_ends[l]
				tied_edge_probability += expected_transitions[start, end]

			# Update each entry
			for l in xrange( tied_edges[k], tied_edges[k+1] ):
				start = self.tied_edges_starts[l]
				end = self.tied_edges_ends[l]
				expected_transitions[start, end] = tied_edge_probability

		# Calculate the regularizing norm for each node
		for k in xrange( m ):
			for l in xrange( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				norm[k] += expected_transitions[k, li] + \
					transition_pseudocount + \
					self.out_transition_pseudocounts[l] * use_pseudocount

		# For every node, update the transitions appropriately
		for k in xrange( m ):
			# Recalculate each transition out from that node and update
			# the vector of out transitions appropriately
			if norm[k] > 0:
				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					probability = ( expected_transitions[k, li] +
						transition_pseudocount + 
						self.out_transition_pseudocounts[l] * use_pseudocount)\
						/ norm[k]
					self.out_transition_log_probabilities[l] = clog(
						cexp( self.out_transition_log_probabilities[l] ) * 
						edge_inertia + probability * ( 1 - edge_inertia ) )

			# Recalculate each transition in to that node and update the
			# vector of in transitions appropriately 
			for l in xrange( in_edges[k], in_edges[k+1] ):
				li = self.in_transitions[l]
				if norm[li] > 0:
					probability = ( expected_transitions[li, k] +
						transition_pseudocount +
						self.in_transition_pseudocounts[l] * use_pseudocount )\
						/ norm[li]
					self.in_transition_log_probabilities[l] = clog( 
						cexp( self.in_transition_log_probabilities[l] ) *
						edge_inertia + probability * ( 1 - edge_inertia ) )

		visited = numpy.zeros( self.silent_start, dtype=numpy.int32 )
		for k in xrange( self.silent_start ):
			# If this distribution has already been trained because it is tied
			# to an earlier state, don't bother retraining it as that would
			# waste time.
			if visited[k] == 1:
				continue
			
			# Mark that we've visited this state
			visited[k] = 1

			# Mark that we've visited all states in this tied state group.
			for l in xrange( tied_states[k], tied_states[k+1] ):
				li = self.tied[l]
				visited[li] = 1

			# Re-estimate the emission distribution for every non-silent state.
			# Take each emission weighted by the probability that we were in 
			# this state when it came out, given that the model generated the 
			# sequence that the symbol was part of. Take into account tied
			# states by only training that distribution one time, since many
			# states are pointing to the same distribution object.
			self.states[k].distribution.from_summaries( 
				inertia=distribution_inertia )

	cdef void _train_viterbi( self, numpy.ndarray sequences, 
		double transition_pseudocount, int use_pseudocount, 
		double edge_inertia, double distribution_inertia ):
		"""
		Performs a simple viterbi training algorithm. Each sequence is tagged
		using the viterbi algorithm, and both emissions and transitions are
		updated based on the probabilities in the observations.
		"""

		cdef numpy.ndarray sequence
		cdef list sequence_path_pairs = []

		for sequence in sequences:

			# Run the viterbi decoding on each observed sequence
			log_sequence_probability, sequence_path = self.viterbi( sequence )
			if log_sequence_probability == NEGINF:
				print( "Warning: skipped impossible sequence {}".format(sequence) )
				continue

			# Strip off the ID
			for i in xrange( len( sequence_path ) ):
				sequence_path[i] = sequence_path[i][1]

			sequence_path_pairs.append( (sequence, sequence_path) )

		self._train_labelled( sequence_path_pairs, 
			transition_pseudocount, use_pseudocount, edge_inertia, 
			distribution_inertia )

	cdef void _train_labelled( self, list sequences,
		double transition_pseudocount, int use_pseudocount,
		double edge_inertia, double distribution_inertia ):
		"""
		Perform training on a set of sequences where the state path is known,
		thus, labelled. Pass in a list of tuples, where each tuple is of the
		form (sequence, labels).
		"""

		cdef int i, j, m=len(self.states), n, a, b, k, l, li
		cdef numpy.ndarray sequence 
		cdef list labels
		cdef State label
		cdef list symbols = [ [] for i in xrange(m) ]
		cdef int [:] tied_states = self.tied_state_count

		# Define matrices for the transitions between states, and the weight of
		# each emission for each state for training later.
		cdef int [:,:] transition_counts
		transition_counts = numpy.zeros((m,m), dtype=numpy.int32)

		cdef int [:] in_edges = self.in_edge_count
		cdef int [:] out_edges = self.out_edge_count

		# Define a mapping of state objects to index 
		cdef dict indices = { self.states[i]: i for i in xrange( m ) }

		# Keep track of the log score across all sequences 
		for sequence, labels in sequences:
			n = len(sequence)

			# Keep track of the number of transitions from one state to another
			transition_counts[ self.start_index, indices[labels[0]] ] += 1
			for i in xrange( len(labels)-1 ):
				a = indices[labels[i]]
				b = indices[labels[i+1]]
				transition_counts[ a, b ] += 1
			transition_counts[ indices[labels[-1]], self.end_index ] += 1

			# Indicate whether or not an emission came from a state or not.
			i = 0
			for label in labels:
				if label.is_silent():
					continue
				
				# Add the symbol to the list of symbols emitted from a given
				# state.
				k = indices[label]
				symbols[k].append( sequence[i] )

				# Also add the symbol to the list of symbols emitted from any
				# tied states to the current state.
				for l in xrange( tied_states[k], tied_states[k+1] ):
					li = self.tied[l]
					symbols[li].append( sequence[i] )

				# Move to the next observation.
				i += 1

		cdef double [:] norm = numpy.zeros( m )
		cdef double probability

		cdef int [:] tied_edges = self.tied_edge_group_size
		cdef int tied_edge_probability 
		# Go through the tied state groups and add transitions from each member
		# in the group to the other members of the group.
		# For each group defined.
		for k in xrange( len( tied_edges )-1 ):
			tied_edge_probability = 0

			# For edge in this group, get the sum of the edges
			for l in xrange( tied_edges[k], tied_edges[k+1] ):
				start = self.tied_edges_starts[l]
				end = self.tied_edges_ends[l]
				tied_edge_probability += transition_counts[start, end]

			# Update each entry
			for l in xrange( tied_edges[k], tied_edges[k+1] ):
				start = self.tied_edges_starts[l]
				end = self.tied_edges_ends[l]
				transition_counts[start, end] = tied_edge_probability

		# Calculate the regularizing norm for each node for normalizing the
		# transition probabilities.
		for k in xrange( m ):
			for l in xrange( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				norm[k] += transition_counts[k, li] + transition_pseudocount +\
					self.out_transition_pseudocounts[l] * use_pseudocount

		# For every node, update the transitions appropriately
		for k in xrange( m ):
			# Recalculate each transition out from that node and update
			# the vector of out transitions appropriately
			if norm[k] > 0:
				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					probability = ( transition_counts[k, li] +
						transition_pseudocount + 
						self.out_transition_pseudocounts[l] * use_pseudocount)\
						/ norm[k]
					self.out_transition_log_probabilities[l] = clog(
						cexp( self.out_transition_log_probabilities[l] ) * 
						edge_inertia + probability * ( 1 - edge_inertia ) )

			# Recalculate each transition in to that node and update the
			# vector of in transitions appropriately 
			for l in xrange( in_edges[k], in_edges[k+1] ):
				li = self.in_transitions[l]
				if norm[li] > 0:
					probability = ( transition_counts[li, k] +
						transition_pseudocount +
						self.in_transition_pseudocounts[l] * use_pseudocount )\
						/ norm[li]
					self.in_transition_log_probabilities[l] = clog( 
						cexp( self.in_transition_log_probabilities[l] ) *
						edge_inertia + probability * ( 1 - edge_inertia ) )

		cdef int [:] visited = numpy.zeros( self.silent_start,
			dtype=numpy.int32 )

		for k in xrange( self.silent_start ):
			# If this distribution has already been trained because it is tied
			# to an earlier state, don't bother retraining it as that would
			# waste time.
			if visited[k] == 1:
				continue
			visited[k] = 1

			# We only want to train each distribution object once, and so we
			# don't want to visit states where the distribution has already
			# been retrained.
			for l in xrange( tied_states[k], tied_states[k+1] ):
				li = self.tied[l]
				visited[li] = 1

			# Now train this distribution on the symbols collected. If there
			# are tied states, this will be done once per set of tied states
			# in order to save time.
			self.states[k].distribution.from_sample( symbols[k], 
				inertia=distribution_inertia )
