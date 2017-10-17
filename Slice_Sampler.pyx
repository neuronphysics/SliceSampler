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
from libc.stdint cimport *
cdef extern from "math.h":
     cdef double INFINITY
     cdef double NAN
     cdef int isinf(double) nogil
 
from libcpp.vector cimport vector
"""
****************************      slice sampling      ****************************    
Reference: Neal, R.M. (2003). Slice sampling. The Annals of Statistics 31, 705-767.
**********************************************************************************

"""

cdef extern from "<math.h>" nogil:
     cdef double floor(double)
     cdef double log(double)
     cdef double pow(double, double)
cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t  
    

###############################################################################################################################################
#######################                             Import random number generators from GSL                            #######################
###source-https://github.com/UT-Python-Fall-2013/Class-Projects/blob/3b759f8b92bd141c0eb644661db647e63e09b4a7/pope_project/poyla_sampler.pyx###
#######################                                                                                                 #######################
###############################################################################################################################################
cdef extern from "gsl/gsl_rng.h":#nogil:
     ctypedef struct gsl_rng_type:
        pass
     ctypedef struct gsl_rng:
        pass
     gsl_rng_type *gsl_rng_mt19937
     gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
  
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef extern from "gsl/gsl_randist.h":#nogil:
     double unif "gsl_rng_uniform"(gsl_rng * r)
     double unif_interval "gsl_ran_flat"(gsl_rng * r,double,double)	## syntax; (seed, lower, upper)
     double exponential "gsl_ran_exponential"(gsl_rng * r,double) ## syntax; (seed, mean) ... mean is 1/rate
        
ctypedef double (*func_t)(double)

cdef class wrapper:
    cdef func_t wrapped
    def __call__(self, value):
        return self.wrapped(value)
    def __unsafe_set(self, ptr):
        self.wrapped = <func_t><void *><size_t>ptr   
        


cdef void stepping_out(double* L, double *R, double* g_interv, double* x0, double* y,
                       double* w, int* m, double* bound, int* is_bound, func_t f):
     """
     Function for finding an interval around the current point 
     using the "stepping out" procedure (Figure 3, Neal (2003))
     Parameters of stepping_out subroutine:
        Input:
            x0 ------------ the current point
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
             m ------------ integer limiting the size of a slice to "m*w"
      bound[2] ------------ bounds for the distribution of x (if any)
   is_bound[2] ------------ indicators whether there are bounds on left and right
     (*func_t) ------------ routine to compute g(x) = log(f(x))
       Output:
            L ------------- the left side of found interval
            R ------------- the right side of found interval
  g_interv[2] ------------- log(f(x)) evaluated in the limits of the interval (if there are no bounds)
     """
     cdef double u
     cdef J, K
     
     #Initial guess for the interval
     u = unif_interval(r,0,1)
     L[0] = x0[0] - w[0]*u
     R[0] = L[0] + w[0]
     
     #Get numbers of steps tried to left and to right
     u = unif_interval(r,0,1)
     J = <uint64_t>floor(m[0]*u)
     K = (m[0]-1)-J
     
     #Initial evaluation of g in the left and right limits of the interval 
     g_interv[0]=f(L[0])
     g_interv[1]=f(R[0])
     
     #Step to left until leaving the slice 
     while ((J > 0) and (g_interv[0] > y[0])):
           L[0] -= w[0]
           g_interv[0]=f(L[0])
           J-=1
  

     #Step to right until leaving the slice */
     while ((K > 0) and (g_interv[1] > y[0])):
           R[0] += w[0]
           g_interv[1]=f(R[0])
           K-=1

     if (is_bound[0] and (L[0] <= bound[0])):
        L[0] = bound[0]     # g_interv[0] is already equal to -FLT_MAX  
     if (is_bound[1] and (R[0] >= bound[1])):
        R[0] = bound[1]     # g_interv[1] is already equal to -FLT_MAX  

     return
     
cdef void doubling(double* L, double* R, double* g_interv, double* x0, double* y,
                   double* w, int* p, double* bound,  int* is_bound,  int* unimodal,
                   func_t f):
     """
     Function for finding an interval around the current point
     using the "doubling" procedure (Figure 4, Neal (2003))
         Input:
            x0 ------------ the current point
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
             p ------------ integer limiting the size of a slice to "2^p*w"
      bound[2] ------------ bounds for the distribution of x (if any)
   is_bound[2] ------------ indicators whether there are bounds on left and right
      unimodal ------------ indicator whether the distribution at question is unimodal or not
     (*func_t) ------------ routine to compute g(x) = log(f(x))
       Output:
            L ------------- the left side of found interval
            R ------------- the right side of found interval
  g_interv[2] ------------- log(f(x)) evaluated in the limits of the interval (if there are no bounds)
    
     """
     cdef double u
     cdef int K
     cdef bint go_left, go_right, now_left
     
     
     #Initial guess for the interval
     u = unif_interval(r,0,1)
     L[0] = x0[0] - w[0]*u
     R[0] = L[0] + w[0]

     K = p[0]
     go_left = True
     go_right = True

     # Initial evaluation of g in the left and right limits of the interval 
     g_interv[0]= f(L[0])
     g_interv[1]= f(R[0])
     if (is_bound[0] and (L[0] <= bound[0])):
        go_left = False
     if (is_bound[1] and (R[0] >= bound[1])):
        go_right = False
     #leave the value of L or R outside the range to be able to perform back-tracking
     if (unimodal[0]):
        if (g_interv[0] <= y[0]):
           go_left = False            #left limit is already outside the slice  
        if (g_interv[1] <= y[0]):
           go_right = False           #right limit is already outside the slice 
  
     if ((not go_left) and (not go_right)):
         K = 0
     # Perform doubling until both ends are outside the slice 
     while ((K > 0) and ((g_interv[0] > y[0]) or (g_interv[1] > y[0]))):
           if (go_right and go_left):    # we have to decide where to go in this step 
              u = unif_interval(r,0,1)
              now_left = (u < 0.5)
           else:
             if (not go_right):
                now_left = True
             else:
                now_left = False

           if (now_left):
              L[0] -= (R[0] - L[0])
              g_interv[0]=f(L[0])
              if (is_bound[0] and (L[0] <= bound[0])):
                 go_left = False
              if ((unimodal[0]) and (g_interv[0] <= y[0])):     
                 go_left = False  #if unimodal distribution left limit is already outside the slice, no slice points to the left 
           else:
              R[0] += (R[0] - L[0])
              g_interv[1]=f(R[0])
              if (is_bound[1] and (R[0] >= bound[1])): 
                 go_right = False
              if ((unimodal[0]) and (g_interv[1] <= y[0])):
                 go_right = False  #if unimodal distribution right limit is already outside the slice, no slice points to the right 
           K-=1
           if ((not go_left) and (not go_right)):
              K = 0
     return
     
cdef void accept_doubling(int* accept, double* x0, double* x1, double* y, double* w, double* L, double* R, func_t f):
     """
     Acceptance test of newly sampled point when the "doubling" procedure has been used to find an 
     interval to sample from (Figure 6, Neal (2003))
     Parameters
         Input:
            x0 ------------ the current point
            x1 ------------- the possible next candidate point
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
             L ------------ the left side of found interval
             R ------------ the right side of found interval
             p ------------ integer limiting the size of a slice to "2^p*w"
     (*func_t) ------------ routine to compute g(x) = log(f(x))
       Output:
        accept ------------ 1/0 indicating whether the point is acceptable or not
     """
     cdef double interv1[2]
     cdef double g_interv1[2]
     cdef bint D
     cdef double w11, mid
     w11 = 1.1*w[0]
     interv1[0] = L[0]
     interv1[1] = R[0]
     D = False
     accept[0] = 1
     while ((accept[0]) and (interv1[1] - interv1[0] > w11)):
           mid = 0.5*(interv1[0] + interv1[1])
           if (x1[0] < mid):
               if (x0[0] >= mid):
                   D = True
               interv1[1] = mid
               g_interv1[1] = f(interv1[1])
           else:
             if (x0[0] < mid):
                D = True
             interv1[0] = mid
             g_interv1[0] = f(interv1[0])
           if (D and (g_interv1[0] <= y[0]) and (g_interv1[1] <= y[0])):
              accept[0] = 0
     return


cdef void shrinkage(double* L, double* R, double* x1, double* g_interv, double* x0, double* y,
                    double* w, int* doubling, int* unimodal, func_t f):
     """
     Function to sample a point from the interval while skrinking the interval when the sampled point is 
     not acceptable (Figure 5, Neal (2003))
         Input:
             L ------------ the left side of found interval
             R ------------ the right side of found interval
   g_interv[2] ------------ log(f(x)) evaluated in the limits of the interval (if there are no bounds)
            x0 ------------ the current point
             y ------------ logarithm of the vertical level defining the slice
             w ------------ estimate of the typical size of a slice
      unimodal ------------ indicator whether the distribution at question is unimodal or not
     (*func_t) ------------ routine to compute g(x) = log(f(x))
      doubling ------------ 0/1 indicating whether doubling was used to find an interval
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
              if (doubling[0] and (not unimodal[0])):
                   accept_doubling(&accept, x0, x1, y, w, L, R, f )
                   if (not accept): # do shrinkage 
                      if (x1[0] < x0[0]):
                         L[0] = x1[0]
                         g_interv[0] = gx1
                      else:
                         R[0] = x1[0]
                         g_interv[1] = gx1
              else:
                  accept = 1
           else:   # do shrinkage 
              if (x1[0] < x0[0]):
                 L[0] = x1[0]
                 g_interv[0] = gx1
              else:
                 R[0] = x1[0]
                 g_interv[1] = gx1
           if (not accept):
              break
     return
 
 
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
cdef double log_beta(double x) nogil:
     #Log of beta distribution with second argument b=1
     cdef double a=5.
     return log(a)+(a-1)*log(x)
     
cdef wrapper make_wrapper(func_t f):
    cdef wrapper W=wrapper()
    W.wrapped=f
    return W

def slice_sampler(int n_sample,
                  wrapper f,                                    
                  double x0_start=0,
                  bint adapt_w=False,
                  int m=None,
                  int p=None,
                  int unimodal=1,
                  double w_start=0.1,
                  unsigned char[:] interval_method ='doubling'):
    
     cdef double x0 = x0_start
     cdef double vertical, w   
     cdef double L, R, interval_length, x1   
     cdef int doubling_used=1
     if (interval_method!='doubling'):
        doubling_used=0
     cdef double w_n=1
     cdef np.ndarray[ndim=1, dtype=np.float64_t] g_interv,bound
     cdef np.ndarray[ndim=1, dtype=np.int64_t] is_bound
     is_bound = np.zeros(2, dtype=np.int64)
     bound = np.zeros(2, dtype=np.float64)
     g_interv = np.zeros(2, dtype=np.float64)
     cdef vector[double] samples #http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
     w=w_start
     cdef Py_ssize_t i
     cdef int accept
     L = 0.
     R = 0.
     bound[0] = -INFINITY 
     bound[1] = INFINITY
     for 0<= i <n_sample:
              vertical = f(x0) - exponential(r, 1) 
              if isinf(bound[0]):
                 is_bound[0]=0
                     
              if isinf(bound[1]):
                 is_bound[1]=0

              if (interval_method=='doubling'):
                     
                  doubling(&L, 
                           &R, 
                           &g_interv[0], 
                           &x0, 
                           &vertical,
                           &w, 
                           &p, 
                           &bound[0], 
                           <int *>(&is_bound[0]),
                           &unimodal,
                           f.wrapped)
              elif (interval_method=='stepping'):

                  stepping_out(&L, 
                               &R, 
                               &g_interv[0], 
                               &x0, 
                               &vertical,
                               &w, 
                               &m, 
                               &bound[0], 
                               <int *>(&is_bound[0]), 
                               f.wrapped)     
              else:
                  raise ValueError("%s is not an acceptable interval method for slice sampler"%interval_method )
              shrinkage(&L, 
                        &R, 
                        &x1, 
                        &g_interv[0], 
                        &x0, 
                        &vertical,
                        &w, 
                        &doubling_used, 
                        &unimodal, 
                        f.wrapped)
              samples.push_back(x1) 
              x0=x1
              if adapt_w:
                 interval_length=R-L
                 w=pow(w,w_n/(w_n+1))*pow(interval_length/2.,1./(w_n+1.))
                 w_n+=1.
              print L,R,x0
     return samples 
              