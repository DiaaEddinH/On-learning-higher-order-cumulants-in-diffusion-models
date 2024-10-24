#ifndef _LANGEVIN_H
#define _LANGEVIN_H

double drift(double x);
double diff_coeff(double t, double logsigma);
void em_solver(int traj, int time_steps, double step_size, double *x, double logsigma);

#endif
