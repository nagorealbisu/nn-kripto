// chebyshev.h
#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

double *chebyshev_coefficients ( double a, double b, int n, double f ( double x ) );
double *chebyshev_interpolant ( double a, double b, int n, double c[], int m, double x[] );
double *chebyshev_zeros ( int n );
void timestamp ( void );

double chebyshev_eval(double a, double b, int n, double *c, double x);

#endif