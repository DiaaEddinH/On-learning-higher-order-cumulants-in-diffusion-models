#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void calc_binom(int n, int **C)
{
    for (int i = 0; i<n+1; i++)
    {
        C[i][0] = C[i][i] = 1;
        for (int k=1; k < i; k++)
            C[i][k] = C[i-1][k-1] + C[i-1][k];
    }
}

void calc_moments(int size, int order, double *samples, double *moments)
{
    int i, n;
    double x, powers;
	double avg = 0.;

    for (i=0; i<size; i++)
		avg += samples[i];
	avg /= size;

    for (i=0; i<size; i++)
    {
        x = samples[i];
        powers = 1.;
        for (n=0; n<order; n++)
        {
            powers *= (x - avg);
            moments[n] += powers;
        }
    }

    for (n=0; n < order; n++)
        moments[n] /= (double)size;
}

void calc_cumulants(int order, double *moments, double *cumulants, int **binom)
{
	int n, k;
	cumulants[0] = moments[0];
    for (n=1; n<order; n++)
    {
        cumulants[n] = moments[n];
        for (k=1; k<n; k++)
            cumulants[n] -= binom[n][k] * cumulants[k] * moments[n-k-1];
    }
}

void jackbin_cumulants(int size, int order, int bin_size, double * samples, double * mean_moments, double * moment_err, double * mean_cumulants, double * cumulant_err, int ** binom)
{
	int i, bin;
	int num_bins = size/bin_size;
	double * jack_samples = (double *)malloc((size-bin_size) * sizeof(double));
	double * moments = (double *)malloc(order * sizeof(double));
	double * cumulants = (double *)malloc(order * sizeof(double));

    double **jack_moments = (double **)malloc(num_bins * sizeof(double *));
    double **jack_cumulants = (double **)malloc(num_bins * sizeof(double *));
    for (bin = 0; bin < num_bins; bin++) {
        jack_moments[bin] = (double *)malloc(order * sizeof(double));
        jack_cumulants[bin] = (double *)malloc(order * sizeof(double));
    }

	for (i=0; i < order; i++) {
		mean_moments[i] = 0;
		mean_cumulants[i] = 0;
	}


	for (bin=0; bin < num_bins; bin++) {
		int idx = 0;
		for (i=0; i<size; i++) {
			if (i < bin * bin_size || i >= (bin + 1) * bin_size)
				jack_samples[idx++] = samples[i];
		}

		calc_moments(size-bin_size, order, jack_samples, moments);
		calc_cumulants(order, moments, cumulants, binom);

		for (i=0; i<order; i++) {
			jack_moments[bin][i] = moments[i];
			jack_cumulants[bin][i] = cumulants[i];

			mean_moments[i] += moments[i];
			mean_cumulants[i] += cumulants[i];
		}
	}

    for (i = 0; i < order; i++) {
        mean_moments[i] /= num_bins;
        mean_cumulants[i] /= num_bins;
    }

	for (i = 0; i<order; i++) {
		moment_err[i] = 0;
		cumulant_err[i] = 0;

		for(bin=0; bin<num_bins; bin++) {
			moment_err[i] += (jack_moments[bin][i] - mean_moments[i]) * (jack_moments[bin][i] - mean_moments[i]);
			cumulant_err[i] += (jack_cumulants[bin][i] - mean_cumulants[i]) * (jack_cumulants[bin][i] - mean_cumulants[i]);
		}
		moment_err[i] = sqrt((num_bins - 1) * moment_err[i] / num_bins);
		cumulant_err[i] = sqrt((num_bins - 1) * cumulant_err[i] / num_bins);
	}

	for (i=0; i<num_bins; i++) {
		free(jack_moments[i]);
		free(jack_cumulants[i]);
	}
	free(jack_samples);
	free(moments);
	free(cumulants);
	free(jack_moments);
	free(jack_cumulants);
}
