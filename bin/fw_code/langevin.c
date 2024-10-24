#include <math.h>
#include "rngmit.h"
#include "langevin.h"

double drift(double x) {
    return 0*x;
}

double diff_coeff(double t, double logsigma) {
	return exp(t * logsigma);
}

void em_solver(int traj, int time_steps, double step_size, double *x, double logsigma) {
    double eta;
	int i, step;
	double sqrtstepsize = sqrt(2 * step_size);
	double t;

    for (step=0; step<time_steps; step++)
    {
		t = i * step_size;
        for(i=0; i<traj; i++) {
			gaussian(&eta);
            x[i] = x[i] + diff_coeff(t, logsigma) * sqrtstepsize * eta;
        }
    }
}
