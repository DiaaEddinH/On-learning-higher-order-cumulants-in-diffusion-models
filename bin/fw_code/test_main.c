#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "cumulants.h"

#define Nmoments 8 /*Number of moments*/
#define BIN_SIZE 10000

void write_to_files(double *moments, double *cumulants, FILE *filem, FILE *filec);

int main(int argc, char *argv[]) {
	unsigned int start, end;
    start = clock();

	int i;
	int Ntraj = atoi(argv[1]); /*number of trajectories*/
	char *file_suffix = argv[2]; /*file_suffix e.g. 100K, 1M etc*/

	double moments[Nmoments], cumulants[Nmoments];
	double moment_err[Nmoments], cumulant_err[Nmoments];

	if(argc != 3) {
		fprintf(stderr, "Usage: %s <Ntraj> <time_steps> <file_suffix>\n", argv[0]);
		return EXIT_FAILURE;
	}
	char samples_file[100];
	sprintf(samples_file, "../data/double_peak_samples_%s.dat", file_suffix);
	FILE * fp = fopen(samples_file, "r");

	if (fp == NULL) {
		fprintf(stderr, "Error opening files for reading samples.\n");
		return EXIT_FAILURE;
	}

	char moments_file[100], cumulants_file[100];
	char moments_errfile[100], cumulants_errfile[100];
	sprintf(moments_file, "../data/jack_moments_%s.dat", file_suffix);
	sprintf(cumulants_file, "../data/jack_cumulants_%s.dat", file_suffix);
	sprintf(moments_errfile, "../data/jack_momentserr_%s.dat", file_suffix);
	sprintf(cumulants_errfile, "../data/jack_cumulantserr_%s.dat", file_suffix);

	FILE * fm = fopen(moments_file, "w");
	FILE * fc = fopen(cumulants_file, "w");
	FILE * fm_err = fopen(moments_errfile, "w");
	FILE * fc_err = fopen(cumulants_errfile, "w");

	if (fm == NULL || fc == NULL || fm_err == NULL || fc_err == NULL) {
		fprintf(stderr, "Error opening files for writing.\n");
		return EXIT_FAILURE;
	}

	int **C = (int **)malloc((Nmoments+1) * sizeof(int *));
	for (i = 0; i < Nmoments+1; i++)
		C[i] = (int *)malloc((i+1) * sizeof(int));
	calc_binom(Nmoments, C);

	double *samples = (double *)malloc(Ntraj * sizeof(double));

	int step;

	for (step=0; !feof(fp); step++) {
		if (fscanf(fp, "%lf", &samples[step]) != 1)
			break;

		if ((step + 1) % Ntraj == 0) {
			step=0;
			jackbin_cumulants(Ntraj, Nmoments, BIN_SIZE, samples, moments, moment_err, cumulants, cumulant_err, C);
			write_to_files(moments, cumulants, fm, fc);
			write_to_files(moment_err, cumulant_err, fm_err, fc_err);
		}
	}

	fclose(fm);
	fclose(fc);
	fclose(fm_err);
	fclose(fc_err);
	fclose(fp);

	for(i=0;i<Nmoments+1; i++)
		free(C[i]);
	free(C);
	free(samples);


    end = clock();
    printf("Time taken: %f seconds\n", (float)(end - start) / CLOCKS_PER_SEC);
	return 0;
}

void write_to_files(double *moments, double *cumulants, FILE *filem, FILE *filec)
{
	int i;

    fprintf(filem, "%2.8lf", moments[0]);
	fprintf(filec, "%2.8lf", cumulants[0]);
    for (i=1; i<Nmoments; i++){
		fprintf(filem, ",%2.8lf", moments[i]);
        fprintf(filec, ",%2.8lf", cumulants[i]);
	}
    fprintf(filem, "\n");
	fprintf(filec, "\n");
}
