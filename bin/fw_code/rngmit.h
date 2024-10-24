/*for 32 bit*/
/*
#define rng_conv 2.3283064365387e-10
*/
/*for 64 bit*/

#ifndef _RNGMIT_H
#define _RNGMIT_H

#define rng_conv 5.421010862427522e-20


#define rngmit (rng_conv*(rng_ia[rng_p=rng_mod[rng_p]] += rng_ia[rng_pp=rng_mod[rng_pp]]))
#define rngmitint (rng_ia[rng_p=rng_mod[rng_p]] += rng_ia[rng_pp=rng_mod[rng_pp]])

extern unsigned long int rng_ia[55];
extern int rng_p,rng_pp;
extern int rng_mod[55];

void rngseed(unsigned long int s);
void gaussian2(double *eta1, double *eta2);
void gaussian(double *eta);
void sample_double_peak(int sample_size, double mu, double sigma, double *x);
void sample_normal(int sample_size, double mu, double sigma, double *x);


#endif
