#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rngmit.h"

int main() {
	unsigned int start, end;
    start = clock();
	// Init seed
	rngseed(time(NULL));

	int i;
	int Ntraj = 1e6; /*number of trajectories*/

    char filename[100];
	sprintf(filename, "../data/double_peak_samples_1M.dat");

	FILE * fp = fopen(filename, "w");

	double mu = 1;
	double sigma = 0.25;
	double *x = malloc(Ntraj * sizeof(double));
	sample_double_peak(Ntraj, mu, sigma, x); //Populate with double peak

	for(i=0; i<Ntraj; i++)
		fprintf(fp, "%2.10lf\n", x[i]);

	fclose(fp);
	free(x);

    end = clock();
    printf("Time taken: %f seconds\n", (float)(end-start)/CLOCKS_PER_SEC);
	return 0;
}
