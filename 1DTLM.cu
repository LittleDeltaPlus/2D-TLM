
// 1D TLM for CUDA - Department of EEE, Universiy of Nottingham 2020
//
// Simulates a line divided into NX segments (nodes) of length dl
//
// Origin of line is matched to the source impedance i.e. no reflection from the left side of the source
//
// Line is excited at node 0 with a gaussian voltage
//
// Line is terminated with a short circuit to ground 
// (results in an equal and opposite reflection at the end of the line)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>  // for setprecision
#include <time.h>   // for clock

#define c0 299792458        // speed of light in a vacuum
#define PI 3.1415926589793  // mmmm PI
#define mu0 PI*4e-7         // magnetic permeability in a vacuum H/m
#define eta0 c*mu0          // wave impedance in free space 

using namespace std;

double  tlmSource(double, double, double);          // excitation function

ofstream gaussian_time("gaussian_excitation.out");  // log excitation function to file
ofstream line_voltage("line_voltage_1.out");        // log probe voltage at a pint on the line versus time


// TLM scatter on GPU
__global__ void tlmScatter(double* VR, double* VL, int N, int n, double source)
{
    unsigned int idx = threadIdx.x;
    //apply source
    if (idx == 0)
        VL[0] = source;
    //scatter
    if (idx < N)
    {
        double V = VL[idx] + VR[idx];
        VR[idx] = V - VR[idx];
        VL[idx] = V - VL[idx];
    }
}

//TLM connect and apply boundary on GPU
__global__ void tlmConnect(double* VR, double* VL, double* Vp, int N, int n)
{
    unsigned int idx = threadIdx.x;

    //Connect
    if (idx > 0 && idx < N)
    {
        double V = VR[idx - 1];
        VR[idx - 1] = VL[idx];
        VL[idx] = V;
    }

    //apply boundaries

    if (idx == 0)
    {
        VR[N - 1] *= -1.f;
        VL[0] = 0.f;

        Vp[n] = VL[2] + VR[2]; // o/p
    }
}

int main()
{
    clock_t start, end;
    double cpu_time;

    int NX = 500;   // number of nodes
    int NT = 2000;   // number of time steps

    double dl = 0.05;       // set node line segment length in metres
    double dt = dl / c0;    // set time step duration

    int monitor_point = 100;// location of voltage probe

    // host arrays
    double* VL;              // voltages to the left of the node
    double* VR;              // voltages to the right of the node
    double* Vp;              // voltage probe measurment

    VL = new double[NX];     // allocate storage for VL array
    VR = new double[NX];     // allocate storage for VR array
    Vp = new double[NT];     // allocate storage for Vp array

    // initialise host arrays VL and VR arrays to 0 - i.e. entire line is at 0V
    for (int i = 0; i < NX; i++) {
        VL[i] = 0;
        VR[i] = 0;
        Vp[i] = 0;
    }

    for (int i = 0; i < NT; i++) {
        Vp[i] = 0;
    }

    // device arrays
    double* dev_VL;         // voltages to the left of the node
    double* dev_VR;         // voltages to the right of the node
    double* dev_Vp;         // allocate storage for Vp array

    //allocate memory on device
    cudaMalloc((void**)&dev_VR, NX * sizeof(double));
    cudaMalloc((void**)&dev_VL, NX * sizeof(double));
    cudaMalloc((void**)&dev_Vp, NT * sizeof(double));

    //copy memory areas from host to device
    cudaMemcpy(dev_VR, VR, NX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_VL, VL, NX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Vp, Vp, NT * sizeof(double), cudaMemcpyHostToDevice);

    // set paremeters for gaussian excitation
    double width = 50 * dt;     // gaussian width 
    double delay = 100 * dt;    // set time delay before starting excitation

    // Start of TLM algorithm
    //
    // loop over total time NT in steps of dt

    start = clock();

    for (int n = 0; n < NT; n++)
    {
        double source = tlmSource(n * dt, delay, width);
        tlmScatter <<<1, NX >>> (dev_VR, dev_VL, NX, n, source);
        tlmConnect <<<1, NX >>> (dev_VR, dev_VL, dev_Vp, NX, n);
    }

    // End of TLM algorithm
    
    end = clock();

    // copy array of measured voltages from device
    cudaMemcpy(Vp, dev_Vp, NT * sizeof(double), cudaMemcpyDeviceToHost);

    // free memory allocated on the GPU
    cudaFree(dev_VL);
    cudaFree(dev_VR);
    cudaFree(dev_Vp);

    // write measured voltages to file
    for (int n = 0; n < NT; n++) {
        line_voltage << dt * n << "  " << Vp[n] << endl;
    }

    double TLM_Execution_Time = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by TLM algorithm : " << fixed << TLM_Execution_Time << setprecision(5);
    cout << " sec " << endl;
    return 0;
}

double tlmSource(double time, double delay, double width)
{
    // calculate value of gaussian ecitation voltage at time point
    double source = exp(-1.0 * double(time - delay) * double(time - delay) / (width * width));

    // log value of gaussian voltage to file
    gaussian_time << time << "  " << source << endl; //write source funtion to file, comment out for timing

    return source;
}
