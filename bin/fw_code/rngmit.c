
/* Implementation of the additive random number generator of Mitchell and
 * Moore.  The generator itself is implemented as a macro in the accompanying
 * header file.
 *
 * Mark Newman  8 OCT 96
 */

/* Constants */

#include <math.h>
#include "rngmit.h"
#ifndef PI
    #define PI acos(-1.)
#endif

#define a 2416
#define c 374441
#define m 1771875
#define conv 2423.96743336861

/* Globals */

unsigned long int i;
unsigned long int rng_ia[55];
int rng_p,rng_pp;
int rng_mod[55];


void rngseed(unsigned long int s)

/* Function to seed all the random number generators */

{
  int n;

  /* First seed the linear congruential generator */

  i = s;

  /* Use that to seed the additive generator.  Also setup the mod array */

  for (n=0; n<55; n++) {
    rng_ia[n] = conv*(i=(a*i+c)%m);
    rng_mod[n] = n-1;
  }
  rng_mod[0] = 54;

  rng_p = 0;
  rng_pp = 24;

  /* Run off ten thousand random numbers, just to get things started */

  for (n=0; n<10000; n++) rng_ia[rng_p=rng_mod[rng_p]] += rng_ia[rng_pp=rng_mod[rng_pp]];
}

void gaussian(double *eta)
{
  double r, angle, ampli=-2.;		/*distribution=exp(-eta^2/2)*/

  r=sqrt(ampli*log(1.0-rngmit));
  angle= 2 * PI * rngmit;
  *eta=r*cos(angle);
  // *eta2=r*sin(angle);
}

void gaussian2(double *eta1, double *eta2)
{
  double r, angle, ampli=-2.;		/*distribution=exp(-eta^2/2)*/

  r=sqrt(ampli*log(1.0-rngmit));
  angle= 2 * PI * rngmit;
  *eta1=r*cos(angle);
  *eta2=r*sin(angle);
}

void sample_double_peak(int sample_size, double mu, double sigma, double *x) {

	int i;
	double eta1, eta2;

	for(i=0; i<sample_size; i+=2) {
		gaussian2(&eta1, &eta2);
		x[i] = mu + sigma * eta1;
		x[i+1] = -mu + sigma * eta2;
	}
}

void sample_normal(int sample_size, double mu, double sigma, double *x) {

	int i;
	double eta1, eta2;

	for(i=0; i<sample_size; i+=2) {
		gaussian2(&eta1, &eta2);
		x[i] = mu + sigma * eta1;
		x[i+1] = mu + sigma * eta2;
	}
}
