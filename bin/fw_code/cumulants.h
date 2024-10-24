#ifndef _CUMULANTS_H
#define _CUMULANTS_H

double drift(double x);

void calc_binom(int n, int **C);
void calc_moments(int size, int order, double *samples, double *moments);
void calc_cumulants(int order, double *moments, double *cumulants, int **binom);

void calc_boot_moments(int order, double *samples, double *moments);
void calc_boot_cumulants(int order, double * samples, double * moments, double * cumulants, int ** binom);

void jackbin_cumulants(int size, int order, int bin_size, double * samples, double * mean_moments, double * moment_err, double * mean_cumulants, double * cumulant_err, int ** binom);
void mpi_jackbin_cumulants(int size, int order, int bin_size, double * samples, double * mean_moments, double * moment_err, double * mean_cumulants, double * cumulant_err, int ** binom, int rank, int num_procs);

#endif
