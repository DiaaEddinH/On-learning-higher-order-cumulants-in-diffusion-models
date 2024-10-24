#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "cumulants.h"
#include "rngmit.h"
#include "langevin.h"

#define Nmoments 6 /*Number of moments*/
#define LOGSIGMA log(10)

void write_to_files(double *moments, double *cumulants, FILE *filem, FILE *filec);

int main(int argc, char *argv[]) {
	unsigned int start, end;
    start = clock();

	int i, step;
	int Ntraj = atoi(argv[1]); /*number of trajectories*/
	int time_steps = atoi(argv[2]); /*number of time steps*/
	char *file_suffix = argv[3]; /*file_suffix e.g. 100K, 1M etc*/

	int Nmeas = time_steps / 1000; /*number of measurements*/
	double mu = 1;
	double sigma = 0.25;

    double step_size = 1. / (double)time_steps;
    double eta, t;
	double sqrtstepsize = sqrt(2 * step_size);
	double moments[Nmoments], cumulants[Nmoments];

	MPI_Init(&argc, &argv);

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if(argc != 4) {
		if (world_rank == 0) {
			fprintf(stderr, "Usage: %s <Ntraj> <time_steps> <file_suffix>\n", argv[0]);
		}
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	int traj_per_proc = Ntraj / world_size;
	int remainder = Ntraj % world_size;
	if (world_rank < remainder) traj_per_proc++;

	// Init seed per process
	rngseed(time(NULL) + world_rank);

	FILE * fm = NULL;
	FILE * fc = NULL;

	if (world_rank == 0) {
		char moments_file[100], cumulants_file[100];
		sprintf(moments_file, "../data/double_peak_moments_%s.dat", file_suffix);
		sprintf(cumulants_file, "../data/double_peak_cumulants_%s.dat", file_suffix);

		fm = fopen(moments_file, "w");
		fc = fopen(cumulants_file, "w");

		if (fm == NULL || fc == NULL) {
			fprintf(stderr, "Error opening files for writing.\n");
			MPI_Finalize();
			return EXIT_FAILURE;
    	}
	}

	int **C = (int **)malloc((Nmoments+1) * sizeof(int *));
	for (i = 0; i < Nmoments+1; i++)
		C[i] = (int *)malloc((i+1) * sizeof(int));
	calc_binom(Nmoments, C);

	double *x = malloc(traj_per_proc * sizeof(double));
	sample_double_peak(traj_per_proc, mu, sigma, x); //Populate with double peak

	double *all_x = NULL;
	if(world_rank==0)
		all_x = malloc(Ntraj * sizeof(double));

    for (step=0; step<time_steps+1; step++)
    {
		t = step * step_size;
		double coeff = diff_coeff(t, LOGSIGMA);
        for(i=0; i<traj_per_proc; i++) {
			gaussian(&eta);
            x[i] = x[i] + coeff * sqrtstepsize * eta;
		}

		if (step % Nmeas == 0) {
			MPI_Gather(x, traj_per_proc, MPI_DOUBLE, all_x, traj_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			if (world_rank == 0) {
				calc_moments(Ntraj, Nmoments, all_x, moments);
				calc_cumulants(Nmoments, moments, cumulants, C);
				write_to_files(moments, cumulants, fm, fc);
			}
		}
    }

	if (world_rank == 0) {
		fclose(fm);
		fclose(fc);
		free(all_x);
	}

	for(i=0;i<Nmoments+1; i++)
		free(C[i]);
	free(C);
	free(x);

	MPI_Finalize();

    end = clock();
    if (world_rank == 0) {
        printf("Time taken: %f seconds\n", (float)(end - start) / CLOCKS_PER_SEC);
    }
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
