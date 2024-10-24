#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "rngmit.h"

#define LOGSIGMA log(10)

void write_samples(double *samples, int size, FILE *fp);

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
	double sqrtstepsize = sqrt(step_size);

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

	FILE * fp = NULL;

	if (world_rank == 0) {
		char samples_file[100];
		sprintf(samples_file, "../data/double_peak_samples_%s.dat", file_suffix);

		fp = fopen(samples_file, "w");

		if (fp == NULL) {
			fprintf(stderr, "Error opening files for writing.\n");
			MPI_Finalize();
			return EXIT_FAILURE;
    	}
	}

	double *x = malloc(traj_per_proc * sizeof(double));
	sample_double_peak(traj_per_proc, mu, sigma, x); //Populate with double peak

	double *all_x = NULL;
	if(world_rank==0)
		all_x = malloc(Ntraj * sizeof(double));

    for (step=0; step<time_steps+1; step++)
    {
		t = step * step_size;
		double coeff = exp(t * LOGSIGMA);
        for(i=0; i<traj_per_proc; i++) {
			gaussian(&eta);
            x[i] = x[i] + coeff * sqrtstepsize * eta;
		}

		if (step % Nmeas == 0) {
			MPI_Gather(x, traj_per_proc, MPI_DOUBLE, all_x, traj_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			if (world_rank == 0)
				write_samples(all_x, Ntraj, fp);
		}
    }

	if (world_rank == 0) {
		fclose(fp);
		free(all_x);
	}

	free(x);
	MPI_Finalize();

    end = clock();
    if (world_rank == 0) {
        printf("Time taken: %f seconds\n", (float)(end - start) / CLOCKS_PER_SEC);
    }
	return 0;
}

void write_samples(double *samples, int size, FILE *fp)
{
	int i;
    for (i=0; i<size; i++){
		fprintf(fp, "%2.8lf\n", samples[i]);
	}
}
