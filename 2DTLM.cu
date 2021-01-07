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

#include <iostream>
#include <fstream>
#include <ctime>   // for clock
#include <cmath>

#define c 299792458        // speed of light in a vacuum
#define mu0 M_PI*4e-7         // magnetic permeability in a vacuum H/m
#define eta0 c*mu0          // wave impedance in free space

using namespace std;

double  tlmSource(double, double, double);          // excitation function
double ** declare_array2D(int, int);                // Population function

ofstream output("output.out");       // log probe voltage at a pint on the line versus time

struct dev_data{
    double* d_V1;
    double* d_V2;
    double* d_V3;
    double* d_V4;
    const double* coeff;
    double* out;
    const int* d_Ein;
    const int* d_Eout;
};
__global__ void tlmApplySource( dev_data dev,double source, int N){
    //Apply Source
    auto tmp_idx = dev.d_Ein[0] + dev.d_Ein[1] * N;
    dev.d_V1[tmp_idx] = dev.d_V1[tmp_idx] + source;
    dev.d_V2[tmp_idx] = dev.d_V2[tmp_idx] - source;
    dev.d_V3[tmp_idx] = dev.d_V3[tmp_idx] - source;
    dev.d_V4[tmp_idx] = dev.d_V4[tmp_idx] + source;
}

// TLM scatter on GPU
__global__ void tlmScatter(dev_data dev, int N, double source){

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    auto index = idx + idy * N;

//    if (idx == 0 && idy == 0){
//        auto tmp_idx = dev.d_Ein[0] + dev.d_Ein[1] * N;
//        dev.d_V1[tmp_idx] = dev.d_V1[tmp_idx] + source;
//        dev.d_V2[tmp_idx] = dev.d_V2[tmp_idx] - source;
//        dev.d_V3[tmp_idx] = dev.d_V3[tmp_idx] - source;
//        dev.d_V4[tmp_idx] = dev.d_V4[tmp_idx] + source;
//    }

    //scatter
    double Z = dev.coeff[0];
   if ( index < N*N)
    {
        double I = (2 * dev.d_V1[index] + 2 * dev.d_V4[index] - 2 * dev.d_V2[index] - 2 * dev.d_V3[index]) / (4 * Z);
        double V = 2 * dev.d_V1[index] - I * Z;    //port1
        dev.d_V1[index] = V - dev.d_V1[index];
        V = 2 * dev.d_V2[index] + I * Z;         //port2
        dev.d_V2[index] = V - dev.d_V2[index];
        V = 2 * dev.d_V3[index] + I * Z;         //port3
        dev.d_V3[index] = V - dev.d_V3[index];
        V = 2 * dev.d_V4[index] - I * Z;         //port4
        dev.d_V4[index] = V - dev.d_V4[index];
   }
}

//TLM connect and apply boundary on GPU
__global__ void tlmConnect(dev_data dev, int N, int n)
{
    auto idx = blockIdx.x*blockDim.x + threadIdx.x;
    auto idy = blockIdx.y*blockDim.y + threadIdx.y;

    auto index = idx + idy * N;

    //Connect
    if ( idx > 0 && index < N*N)
    {
        auto V = dev.d_V2[index];
        dev.d_V2[index] = dev.d_V4[(idx - 1)+ idy * N];
        dev.d_V4[(idx - 1) + idy * N] = V;
    }

    if ( idy > 0 && index < N*N)
    {
        auto V = dev.d_V1[index];
        dev.d_V1[index] = dev.d_V3[idx + (idy - 1)*N];
        dev.d_V3[idx + (idy - 1)*N] = V;
    }

//    //Apply Boundaries
//    double rXmin = dev.coeff[2];
//    double rXmax = dev.coeff[3];
//    double rYmin = dev.coeff[4];
//    double rYmax = dev.coeff[5];
//
//    if (idy == N-1*N && index < N*N){
//        dev.d_V3[idx + (N - 1)*N] = rYmax * dev.d_V3[idx + (N - 1)*N];
//        dev.d_V1[idx] = rYmin * dev.d_V1[idx];
//    }
//
//    if (idx == n-1 && index < N*N) {
//        dev.d_V4[(N - 1) + idy*N] = rXmax * dev.d_V4[(N - 1) + idy*N];
//        dev.d_V2[idy*N] = rXmin * dev.d_V2[idy*N];
//    }


}

__global__ void applyBoundary(dev_data dev, int N) {
    double rXmin = dev.coeff[2];
    double rXmax = dev.coeff[3];
    double rYmin = dev.coeff[4];
    double rYmax = dev.coeff[5];

    for (int x = 0; x < N; x++) {
        dev.d_V3[x + (N - 1)*N] = rYmax * dev.d_V3[x + (N - 1)*N];
        dev.d_V1[x] = rYmin * dev.d_V1[x];
    }
    for (int y = 0; y < N; y++) {
        dev.d_V4[(N - 1) + y*N] = rXmax * dev.d_V4[(N - 1) + y*N];
        dev.d_V2[y*N] = rXmin * dev.d_V2[y*N];
    }
}

__global__ void evalutateOut(dev_data dev, int N, int n){
    auto tmp_idx = dev.d_Eout[0] + dev.d_Eout[1] * N;
    dev.out[n] = dev.d_V2[tmp_idx] + dev.d_V4[tmp_idx];
}

int main()
{


    clock_t start, end;

    int NX = 100;   // dim one of nodes
    int NY = 100;   // dim 2 of nodes
    int NT = 8192;   // number of time steps
    double dl = 1;       // set node line segment length in metres
    double dt = dl / (sqrt(2.) * c);    // set time step duration


    //2D mesh variables
    double I = 0;
    double** V1 = declare_array2D(NX, NY);
    double** V2 = declare_array2D(NX, NY);
    double** V3 = declare_array2D(NX, NY);
    double** V4 = declare_array2D(NX, NY);
    double v_output[NT];
    for (int n = 0; n < NT; n++){
        v_output[n] = 0;
    }

    double Z = eta0 / sqrt(2.);


    //boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    double coeff[] = {Z, I, rXmin, rXmax, rYmin, rYmax};

    //input / v_output
    double width = 20 * dt * sqrt(2.);
    double delay = 100 * dt * sqrt(2.);
    int Ein[] = { 10,10 };
    int Eout[] = { 15,15 };


    /// device arrays
    double* dev_V1;
    double* dev_V2;
    double* dev_V3;
    double* dev_V4;
    double* dev_coeff;
    double* dev_output;
    int* dev_Ein;
    int* dev_Eout;


    ///allocate memory on device
    auto sz = NX * NY * sizeof(double);
    cudaMalloc((void**)&dev_V1, sz);
    cudaMalloc((void**)&dev_V2, sz);
    cudaMalloc((void**)&dev_V3, sz);
    cudaMalloc((void**)&dev_V4, sz);
    cudaMalloc((void**)&dev_coeff, sizeof(double)*6);
    cudaMalloc((void**)&dev_output, sizeof(double)*NT);
    cudaMalloc((void**)&dev_Ein, sizeof(int)*2);
    cudaMalloc((void**)&dev_Eout, sizeof(int)*2);

    auto err = cudaGetLastError();


    ///copy memory areas from host to device
    cudaMemcpy(dev_V1, V1, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V2, V2, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V3, V3, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V4, V4, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_coeff, coeff, sizeof(double)*6, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Ein, Ein, sizeof(int)*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Eout, Eout, sizeof(int)*2, cudaMemcpyHostToDevice);

    err = cudaGetLastError();


    // Start of TLM algorithm
    //
    // loop over total time NT in steps of dt

    dev_data dev_Data{dev_V1, dev_V2, dev_V3, dev_V4, dev_coeff, dev_output, dev_Ein, dev_Eout};

    start = clock();

    dim3 dimBlock(10,10);
    dim3 dimGrid(ceil(NX/dimBlock.x),ceil(NY/dimBlock.y));

    err = cudaGetLastError();
    int i = 0;
    for (int n = 0; n < NT; n++)
    {

        double source = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
        tlmApplySource <<<1, 1>>> (dev_Data, source, NX);
        err = cudaGetLastError();
        i = 0;
        tlmScatter <<<dimGrid, dimBlock>>> (dev_Data, NX, source);
        err = cudaGetLastError();
        i = 0;
        cudaDeviceSynchronize();
        tlmConnect <<<dimGrid, dimBlock>>> (dev_Data, NX, n);
        err = cudaGetLastError();
        i = 0;
        applyBoundary<<<1, 1>>> (dev_Data, NX);
        err = cudaGetLastError();
        i=0;
        evalutateOut<<<1, 1>>>(dev_Data, NX,n);
        err = cudaGetLastError();
        i = 0;
    }
    err = cudaGetLastError();
    cudaMemcpy(v_output, dev_output, sizeof(double)*NT, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    for (int n = 0; n < NT; n++){
        output << n * dt << "  " <<  v_output[n] << endl;
    }
    // End of TLM algorithm
    
    end = clock();

    // copy array of measured voltages from device

    // free memory allocated on the GPU
    cudaFree(dev_V1);
    cudaFree(dev_V2);
    cudaFree(dev_V3);
    cudaFree(dev_V4);
    cudaFree(dev_output);
    cudaFree(dev_coeff);
    cudaFree(dev_Ein);
    cudaFree(dev_Eout);


//    double TLM_Execution_Time = double(end - start) / double(CLOCKS_PER_SEC);
//    cout << "Time taken by TLM algorithm : " << fixed << TLM_Execution_Time << setprecision(5);
//    cout << " sec " << endl;
//    return 0;
}

double tlmSource(double time, double delay, double width)
{
    // calculate value of gaussian ecitation voltage at time point
    //E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
    double source = exp(-1.0 * double(time - delay) * double(time - delay) / (width * width));

    // log value of gaussian voltage to file
//    gaussian_time << time << "  " << source << endl; //write source funtion to file, comment out for timing

    return source;
}

double** declare_array2D(int NX, int NY) {
    auto** V = new double* [NX];
    for (int x = 0; x < NX; x++) {
        V[x] = new double[NY];
    }

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            V[x][y] = 0;
        }
    }
    return V;
}