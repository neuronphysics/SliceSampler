#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
from cpython cimport array
import cython
import numpy as np
import ctypes
cimport numpy as np
cimport cython
cimport python_unicode

from cython.view cimport memoryview, array  
from libcpp.vector cimport vector

#****************************************************************************************************************************************** 
#************************************************              slice sampling              ************************************************        
#                      Reference: Neal, R.M. (2003). Slice sampling. The Annals of Statistics 31, 705-767.
#****************************************************************************************************************************************** 
#****************************************************************************************************************************************** 
cdef extern from "math.h":
     cdef double INFINITY
     cdef double NAN

cdef extern from "<math.h>":
     cdef double floor(double)
     cdef double log(double)
     cdef double pow(double, double)
     cdef double tgamma(double) 
     double fabs(double)
     
     
cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t  
    

#source-https://github.com/UT-Python-Fall-2013/Class-Projects/blob/3b759f8b92bd141c0eb644661db647e63e09b4a7/pope_project/poyla_sampler.pyx
cdef extern from "gsl/gsl_rng.h":#nogil:
     ctypedef struct gsl_rng_type:
        pass
     ctypedef struct gsl_rng:
        pass
     gsl_rng_type *gsl_rng_mt19937
     gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
  
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef extern from "gsl/gsl_randist.h" nogil:
     double unif "gsl_rng_uniform"(gsl_rng * r)
     double unif_interval "gsl_ran_flat"(gsl_rng * r,double,double)	## syntax; (seed, lower, upper)
     double exponential "gsl_ran_exponential"(gsl_rng * r,double) ## syntax; (seed, mean) ... mean is 1/rate
        
# define a global name for whatever char type is used in the module
ctypedef double (*func_t)(double)

cdef class wrapper:
    cdef func_t wrapped
    def __call__(self, value):
        return self.wrapped(value)
    def __unsafe_set(self, ptr):
        self.wrapped = <func_t><void *><size_t>ptr   
        
cdef bint _isinf(double x):
    return (fabs(x) == INFINITY)

cdef bint _isnan(double x):
    return (x == NAN)

cdef double[::1] stepping_out(double x0, double y, double w, int m, func_t f):
     """
     Function for finding an interval around the current point 
     using the "stepping out" procedure (Figure 3, Neal (2003))
     Parameters of stepping_out subroutine:
        Input:
            x0 ------------ the current point
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
             m ------------ integer limiting the size of a slice to "m*w"
     (*func_t) ------------ routine to compute g(x) = log(f(x))
       Output:
     interv[2] ------------ the left and right sides of found interval
     """
     cdef double[::1] interv =array((2,), itemsize=sizeof(double), format='d')
     cdef double u
     cdef int J, K
     cdef double g_interv[2]
     #Initial guess for the interval
     u = unif_interval(r,0,1)
     
     interv[0] = x0 - w*u
     interv[1] = interv[0]+ w
     
     #Get numbers of steps tried to left and to right
     if m>0:
        u = unif_interval(r,0,1)
        J = <uint64_t>floor(m*u)
        K = (m-1)-J
     
     #Initial evaluation of g in the left and right limits of the interval 
     g_interv[0]=f(interv[0])
     g_interv[1]=f(interv[1])
     
     #Step to left until leaving the slice 
     while (g_interv[0] > y):
           interv[0] -= w
           g_interv[0]=f(interv[0])
           if m>0:
              J-=1
              if (J<= 0):
                 break
  

     #Step to right until leaving the slice */
     while (g_interv[1] > y):
           interv[1] += w
           g_interv[1]=f(interv[1])
           if m>0:
              K-=1
              if (K<= 0):
                 break
     return interv
     
cdef double[::1] doubling(double x0, double y, double w, int p, func_t f):
     """
     Function for finding an interval around the current point
     using the "doubling" procedure (Figure 4, Neal (2003))
         Input:
            x0 ------------ the current point
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
             p ------------ integer limiting the size of a slice to "2^p*w"
     (*func_t) ------------ routine to compute g(x) = log(f(x))
       Output:
     interv[2] ------------ the left and right sides of found interval    
     """
     cdef double[::1] interv =array((2,), itemsize=sizeof(double), format='d')
     cdef double u
     cdef int K     
     cdef bint now_left
     cdef double g_interv[2]
     #Initial guess for the interval
     u = unif_interval(r,0,1)
     interv[0] = x0 - w*u
     interv[1] = interv[0]+ w
     
     if p>0:
        K = p

     # Initial evaluation of g in the left and right limits of the interval 
     g_interv[0]= f(interv[0])
     g_interv[1]= f(interv[1])

     # Perform doubling until both ends are outside the slice 
     while ((g_interv[0] > y) or (g_interv[1] > y)):
                    u = unif_interval(r,0,1)              
                    now_left = (u < 0.5)           
                    if (now_left):
                       interv[0] -= (interv[1] - interv[0])
                       g_interv[0]=f(interv[0])
                    else:
                       interv[1] += (interv[1] - interv[0])
                       g_interv[1]=f(interv[1])
                    
                    if p>0:
                       K -= 1
                       if (K<=0):
                           break
     return interv
     
cdef bint accept_doubling(double x0, double x1, double y, double w, np.ndarray[ndim=1, dtype=np.float64_t] interv, func_t f):
     """
     Acceptance test of newly sampled point when the "doubling" procedure has been used to find an 
     interval to sample from (Figure 6, Neal (2003))
     Parameters
         Input:
            x0 ------------ the current point
            x1 ------------- the possible next candidate point
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
     interv[2] ------------ the left and right sides of found interval    
     (*func_t) ------------ routine to compute g(x) = log(f(x))
       Output:
        accept ------------ True/False indicating whether the point is acceptable or not
     """
     cdef double interv1[2]
     cdef double g_interv1[2]
     cdef bint D
     cdef double w11, mid
     w11 = 1.1*w
     interv1[0] = interv[0]
     interv1[1] = interv[1]
     D = False
     while ( (interv1[1] - interv1[0]) > w11):
           mid = 0.5*(interv1[0] + interv1[1])
           if ((x0 < mid) and (x1 >= mid)) or ((x0 >= mid) and (x1 < mid)):
               D = True
           if (x1 < mid):    
               interv1[1] = mid
               g_interv1[1] = f(interv1[1])
           else:
             interv1[0] = mid
             g_interv1[0] = f(interv1[0])
           if (D and (g_interv1[0] < y) and (g_interv1[1] <= y)):
              return False
     return True


cdef double shrinkage(double x0, double y, double w, np.ndarray[ndim=1, dtype=np.float64_t] interv, bint doubling, func_t f):
     """
     Function to sample a point from the interval while skrinking the interval when the sampled point is 
     not acceptable (Figure 5, Neal (2003))
         Input:
            x0 ------------ the current point
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
     interv[2] ------------ the left and right sides of found interval    
     (*func_t) ------------ routine to compute g(x) = log(f(x))
      doubling ------------ 0/1 indicating whether doubling was used to find an interval
       Output:
            x1 ------------- newly sampled point
     """
     cdef double u, gx1, x1
     cdef bint accept
     cdef double L_bar, R_bar
     L_bar=interv[0]
     R_bar=interv[1]
     x1 = L_bar + 0.5*(R_bar - L_bar)  
     
     while True:
           u = unif_interval(r,0,1)
           x1 = L_bar + u*(R_bar - L_bar)
           gx1=f(x1)
           if (doubling):
              accept=accept_doubling(x0, x1, y, w, interv, f )
           else:
              accept=True
           
           if ((gx1 > y) and accept):                 
              break
           if (x1 < x0):
              L_bar = x1
           else:
              R_bar = x1
     return x1
 
 
cdef void overrelaxation_bisection(double* x1, double* L, double* R, double* x0, double* y, double* w,
                                   int* a, int* doubling, func_t f):
     """
     Overrelaxed update using a bisection method (Figure 10, Neal (2003))
          This procedure will work only with UNIMODAL densities!!!
     Parameters
         Input:
             L ------------ the left side of found interval
             R ------------ the right side of found interval
            x0 ------------ the current point
             a ------------ integer limiting endpoint accuracy to 2^{-a}*w
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
     (*func_t) ------------ routine to compute g(x) = log(f(x))
      doubling ------------ 0/1 indicating whether doubling was used to find an interval
       Output:
            x1 ------------- newly sampled point
          
     """                         
     cdef double mid, w_bar, g_mid
     cdef int a_bar
     cdef bint go_on, go_left, go_right
     cdef double interv_hat[2]
     #When the interval is only of size w, narrow it until the midpoint is inside the slice (or accuracy limit is reached
     w_bar = w[0]
     a_bar = a[0]
     if ((R[0] - L[0]) < 1.1*w[0]):
        go_on = True
        while (go_on):
              mid = 0.5*(L[0] + R[0])
              g_mid=f(mid)
              if ((a_bar == 0) or (g_mid > y[0])):
                 go_on = False
              else:
                 if (x0[0] > mid):
                    L[0] = mid
                 else:
                    R[0] = mid
                 a_bar-=1
                 w_bar *= 0.5
     #Redefine endpoint locations by bisection, to the specified accuracy 
     interv_hat[0] = L[0]
     interv_hat[1] = R[0]
     go_left  = True
     go_right = True
     while ((a_bar > 0) and (go_left or go_right)):
           a_bar-=1
           w_bar *= 0.5
           # Bisection on the left 
           if (go_left):
               mid = interv_hat[0] + w_bar
               g_mid=f(mid)
               if (g_mid <= y[0]): 
                   interv_hat[0] = mid
               else:
                   go_left = False
           # Bisection on the right 
           if (go_right):
              mid = interv_hat[1] - w_bar
              g_mid=f(mid)
              if (g_mid <= y[0]):
                 interv_hat[1] = mid
              else:
                 go_right = False
     # Find a candidate point by flipping from the current point to the opposite side of (hat{L}, hat{R}), 
     # then test for acceptability                                                                         
     x1[0] = interv_hat[0] + interv_hat[1] - x0[0]    # = (L+R)/2 + ((L+R)/2 - x)  
     g_mid=f(x1[0])
     if (g_mid <= y[0]):
        x1[0] = x0[0]
     return

cdef void exact_sampler(double* x1, double* L, double* R, double* g_interv,  double* x0,
                        double* y, func_t f):
     """
     Function to sample a point from the interval while skrinking the interval when the sampled point is not acceptable
     version for the case when the density to sample from is UNIMODAL and the slice was found exactly (up to some precision)
     Parameters
         Input:
             L ------------ the left side of found interval
             R ------------ the right side of found interval
            x0 ------------ the current point
             y ------------ logarithm of the vertical level defining the slice
     (*func_t) ------------ routine to compute g(x) = log(f(x))
       Output:
            x1 ------------- newly sampled point
     """
     cdef double u, gx1
     cdef int accept
     accept = 0
     while True:
           u = unif_interval(r,0,1)
           x1[0] = L[0] + u*(R[0] - L[0])
           gx1=f(x1[0])
           if (gx1 > y[0]):
               accept = 1
           else:   
               # do shrinkage 
               if (x1[0] < x0[0]):
                  L[0] = x1[0]
                  g_interv[0] = gx1
               else:
                  R[0] = x1[0]
                  g_interv[1] = gx1
           if (not accept):
              break
     return
#***** ----------------------------------------------------------------------------------------- *****
#*****            g(x)=log(f(x)): Compute log-density of a full conditional distribution         *****
#***** ----------------------------------------------------------------------------------------- *****
cdef double log_beta(double x):
     #Log of beta distribution with second argument b=1
     cdef double a=5.
     cdef double b=1.
     return log(a)+(a-1.)*log(x)+(b-1.)*log(1.-x)     
   
cdef wrapper make_wrapper(func_t f):
    cdef wrapper W=wrapper()
    W.wrapped=f
    return W

def slice_sampler(int n_sample,
                  wrapper f, 
                  int m = 0,
                  int p = 0,
                  double x0_start=0.0, 
                  bint adapt_w=False, 
                  double w_start=0.1,
                  char* interval_method ='doubling'):
     """
     Inputs:
        n_sample ------------ Number of sample points from the given distribution
               f ------------ A log of the function you want to sample and accepts a scalar as an argument (the x) 
        x0_start ------------ An initial value for x
         adapt_w ------------ Whether to adapt w during sampling. Will work in between samplings
         w_start ------------ Starting value for w, necessary if adapting w during sampling
               p ------------ Integer limiting the size of a slice to (2^p)w. If None, then interval can grow without bound
               m ------------ Integer, where maximum size of slice should be mw. If None, then interval can grow without bound
 interval_method ------------ The method for determining the interval at each stage of sampling. Possible values are 'doubling', 'stepping'.
     """
     cdef unicode s= interval_method.decode('UTF-8', 'strict')
     cdef double x0 = x0_start
     cdef double vertical, w, expon   
     cdef double interval_length  
     cdef bint doubling_used=True
     if (s!=u'doubling'):
        doubling_used=False
     cdef double w_n=1.
     cdef vector[double] samples #http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
     w=w_start
     cdef Py_ssize_t i
     cdef np.ndarray[ndim=1, dtype=np.float64_t] interv
     cdef double[::1] vec_view
     for 0<= i <n_sample:
              expon = exponential(r, 1) 
              vertical = f.wrapped(x0) - expon
                            
              if (s=='doubling'):
                  vec_view=doubling(x0, vertical, w, p, f.wrapped)
                  interv= np.asarray(vec_view)
                  
              elif (s==u'stepping'):
                  vec_view=stepping_out(x0, vertical, w, m, f.wrapped)
                  interv= np.asarray(vec_view)
              else:
                  raise ValueError("%s is not an acceptable interval method for slice sampler"%s )
              
              x0=shrinkage(x0, vertical, w, interv, doubling_used, f.wrapped)
              samples.push_back(x0) 
              
              if adapt_w:
                 interval_length=interv[1]-interv[0]
                 w=pow(w,w_n/(w_n+1))*pow(interval_length/2.,1./(w_n+1.))
                 w_n+=1.
              
     return samples    

def run(int n_sample,               
        double x0_start=0., 
        double w_start=2.5):
    wrap_f=make_wrapper(log_beta)    
    return slice_sampler(n_sample, wrap_f,x0_start=x0_start, w_start=w_start)