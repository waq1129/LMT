Z Abstract Sample Object (ZASO) - (ZASO is for zhe abstract sample object)

Basically, ZASO hides the ugly for loop over the samples in your statistical estimators and related optimization algorithms. This is especially useful when pre-processing is necessary in order to transform your original data into higher dimensional feature space.
ZASO is a MATLAB structure that abstracts the complexity of dealing with large feature space too big to fit in memory, that needs to be transformed from a compact representation that fits in memory. The feature space must be a finite dimensional Euclidean space.

It serves a similar role as "iterator" or "foreach" statement in other programming languages, but for specialized sets of data.

== Motivation ==
=== Memory efficiency ===
The spike trains, and stimulus variables are often very sparse in time, while the features are usually represented on a temporal basis which is not sparse. We need to convolve the basis only when necessary for memory efficiency.

=== Multiple time scales ===
Often we need to deal with two time scales: coarse time scale where stimulus or response variables change/measured, and the fine time scale where the spike timings live. The coarse time scale typically range from 10 ms to 200 ms, and the spike timings in 0.1 ms to 2 ms. Therefore, if we represented the data matrix as in the fine time scale, we waste a lot of memory, since the coarse time scale variables are simply repeated.

== Introduction to ZASO ==
Once created, ZASO encapsulates the independent (and dependent) variable in the correct feature space and abstracts unnecessary indexing. The encapsulation depends on the data and feature space at hand, but the usage in an algorithm would only require calling the basic external interface of ZASO.

=== Basic External Interface ===
    zaso.desc: a string containing the description of the data
    zaso.N: number of data points (in the finest time scale), N > 0
    zaso.dimx: dimension of the feature space for the independent variable
    zaso.dimy: dimension of the dependent variable
    zaso.X: @(idx) returns M subindexed samples of x (? x zaso.dimx)
    zaso.Y: @(idx) returns M subindexed samples of y (? x zaso.dimy)
    zaso.fxsum(zaso, @fctx, zasoIdx): computes sum_{i=1}^N fctx(x_i)
    zaso.fysum(zaso, @fcty, zasoIdx): computes sum_{i=1}^N fcty(y_i)
    zaso.fxysum(zaso, @fctxy, zasoIdx): computes sum_{i=1}^N fctxy(x_i, y_i)
    zaso.fx(zaso, @fct, zasoIdx): returns a matrix where z(:,i) = fct(x_i)
    zaso.fy(zaso, @fct, zasoIdx): returns a matrix where z(:,i) = fct(y_i)
    zaso.fxy(zaso, @fct, zasoIdx): returns a matrix where z(:,i) = fct(x_i, y_i)
    zaso.farray(zaso, {@fct_sum}, {@fct_agg}, zasoIdx): returns a cell array
	with corresponding computation from fxysum and fxy for each function
	handle in the corresponding 1D array. The function handles must take 
	both x_i and y_i

==== subindexing ====
    zaso.Nsub: sub-indexing maximum (for raw it is same as N)
    zaso.sub2idx: @(subidx) sub-index to raw index conversion (identity for raw)

zasoIdx is an index sequence for subindexing (optional).
The only requirement of a ZASO is to have the above basic external interface.

fctx must be able to take a block of (M x zaso.dimx) matrix as data, and return a matrix (vector or scalar) that is the sum over M samples, where M <= N. Similarly for fctxy, it should be able to take (M x zaso.dimx), (M x zaso.dimy).

fct for fx,fy,and fxy must return a (M x d) matrix for some fixed d.

In practice, since each function call requires specifying zaso object twice,
it is prone to copy & paste errors where one forgets to change one of the zaso!
Therefore, there are dummy functions that requires only one zaso object to
prevent one from making such mistakes. They are of the form:

    zasoFxsum(zaso, @fct, zasoIdx)
    zasoFysum(zaso, @fct, zasoIdx)
    zasoFxysum(zaso, @fct, zasoIdx)
    zasoFx(zaso, @fct, zasoIdx)
    zasoFy(zaso, @fct, zasoIdx)
    zasoFxy(zaso, @fct, zasoIdx)
    zasoFarray(zaso, {@fct_sum}, {@fct_agg}, zasoIdx)

=== Typical Life of a ZASO ===

    1. Encapsulation: ZASO is created and loaded with samples (data)
    2. ZASO is passed along an algorithm (e.g. for optimization/estimation)
    3. Algorithm computes a statistic in a form of summation over the samples

=== Utility functions ===
First few raw moments often ocurr in estimation of various kind. So we provide
simple utility functions to compute those.

    zasoEX(zaso)
    zasoEY(zaso)
    zasoEXY(zaso)
    zasoEXX(zaso)

Also, for linear models and quadratic models, linear forms and quadratic forms
are essential.

    zasoLinear(zaso, weight): computes sum_i (x_i * weight)
    zasoQuadratic(zaso, weight): computes sum_i (x_i * weight * x_i')

=== miniBatch ===
Mini batch is a useful way of trading-off memory and speed.

zaso.nMiniBatch: (1) how many samples for a mini batch?

=== Subindexing ===
When subindexing is necessary, create a (possibly temporary) child ZASO that indexes the subset. Since MATLAB does not copy memory until it changes, the memory cost of such strategy is minimal.

=== Immutablility and Caching ===
ZASO is a conceptually IMMUTABLE object. Once the data is loaded, it is not supposed to change. But, caching computed results in the structure is allowed. Use a substructure named 'cache' for storing them.

=== Implementation details ===
We do NOT use the every changing MATLAB object-oriented constructions, but rather implement using basic data types and manual rules to deal with the objects. This allows more flexibility and compatibility. But since we cannot enforce alteration of the users, we are in danger of getting hacked easily and improperly used. Make sure everybody reads this documentation to make sure this does not happen.

== Temporal bases for events ==
When the indexing is through time, and the variables are (possibly marked) events, it is most efficient to represent the variables as a sparse matrix binned with the fine time scale. Temporal bases are often used to transform the sparse events to continuous values over time. Precomputed basis functions are convolved with the sparse stimulus on the fly. 
Only if the temporal bases are of finite support, or can be implemented with a recursive filter bank (such as IIR filters, or Gamma filters), we can efficiently store the temporary features space representation in the memory.
TODO: We can use MATLAB's 'filter' to save the state per block/mini-batch and continue without loss.

== Trial structure ==
Complex indexing is necessary when there are trials.
It is convenient to index by trials, while the summation is over time bins within the trials.
