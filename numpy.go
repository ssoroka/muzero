// Go implementations of things I needed from the numpy library.
package main

import (
	"math"
	"math/rand"
	"sync"
)

// translated from numpy.random.Dirichlet.
// see https://raw.githubusercontent.com/numpy/numpy/1d598997543637b9882d1663117b573210a20583/numpy/random/mtrand.pyx
func dirichlet(alpha []float64) []float64 {
	// alpha   = N.atleast_1d(alpha)
	// k       = alpha.size
	k := len(alpha)
	// if n == 1:
	//     val = N.zeros(k)
	val := make([]float64, k)
	//     for i in range(k):
	sum := float64(0)
	for i := 0; i < k; i++ {
		//         val[i]   = sgamma(alpha[i], n)
		val[i] = sgamma(alpha[i])
		sum += val[i]
	}
	//     val /= N.sum(val)
	for i := 0; i < k; i++ {
		val[i] = val[i] / sum
	}
	// else:
	//     val = N.zeros((k, n))
	//     for i in range(k):
	//         val[i]   = sgamma(alpha[i], n)
	//     val /= N.sum(val, axis = 0)
	//     val = val.T
	// return val
	return val
}

// sgamma is the standard gamma distribution
//           x ** (alpha - 1) * math.exp(-x / beta)
// pdf(x) =  --------------------------------------
//             math.gamma(alpha) * beta ** alpha
//
// borrow implementation from https://fossies.org/linux/numpy/numpy/random/src/legacy/legacy-distributions.c#L37
// function legacy_standard_gamma
//
// sgamma aka legacy_standard_gamma
func sgamma(shape float64) float64 {
	var b, c float64
	var U, V, X, Y float64

	if shape == 1.0 {
		return legacyStandardExponential()
	} else if shape == 0.0 {
		return 0.0
	} else if shape < 1.0 {
		for {
			U = legacyDouble()
			V = legacyStandardExponential()
			if U <= 1.0-shape {
				X = math.Pow(U, 1./shape)
				if X <= V {
					return X
				}
			} else {
				Y = -math.Log((1 - U) / shape)
				X = math.Pow(1.0-shape+shape*Y, 1./shape)
				if X <= (V + Y) {
					return X
				}
			}
		}
	} else {
		b = shape - 1.0/3.0
		c = 1.0 / math.Sqrt(9*b)
		for {
			V = -1.0
			for V <= 0.0 {
				X = legacyGauss()
				V = 1.0 + c*X
			}

			V = V * V * V
			U = legacyDouble()
			if U < 1.0-0.0331*(X*X)*(X*X) {
				return (b * V)
			}
			if math.Log(U) < 0.5*X*X+b*(1.-V+math.Log(V)) {
				return (b * V)
			}
		}
	}
}

func legacyDouble() float64 {
	return rand.Float64()
}

func legacyStandardExponential() float64 {
	/* We use -log(1-U) since U is [0, 1) */
	return -math.Log(1.0 - rand.Float64())
}

var (
	hasGauss  bool    = false
	gauss     float64 = 0
	gaussLock sync.Mutex
)

// TODO: write a context-aware version of this that doesn't need to be thread safe and doesn't need locking, or just always do the slow version(?)
func legacyGauss() float64 {
	gaussLock.Lock()
	defer gaussLock.Unlock()
	if hasGauss {
		temp := gauss
		hasGauss = false
		gauss = 0
		return temp
	}
	var f, x1, x2, r2 float64

loop:
	x1 = 2.0*legacyDouble() - 1.0
	x2 = 2.0*legacyDouble() - 1.0
	r2 = x1*x1 + x2*x2
	if r2 >= 1.0 || r2 == 0.0 {
		goto loop
	}

	/* Polar method, a more efficient version of the Box-Muller approach. */
	f = math.Sqrt(-2.0 * math.Log(r2) / r2)
	/* Keep for next call */
	gauss = f * x1
	hasGauss = true
	return f * x2
}
