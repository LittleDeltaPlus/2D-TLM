// 2D TLM for CUDA - Seámus Doran, Universiy of Nottingham 2020
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
#include <chrono>
#include <fstream>
#include <ctime>   // for clock
#include <cmath>

#define c 299792458        // speed of light in a vacuum
#define mu0 M_PI*4e-7         // magnetic permeability in a vacuum H/m
#define eta0 c*mu0          // wave impedance in free space

using namespace std;

double** declare_array2D(int, int);                // Population function

ofstream output("output.out");       // log probe voltage at a pint on the line versus time

/**
 *
 */
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

/**
 *
 * @param dev
 * @param source
 * @param N
 */
__global__ void tlmApplySource( dev_data dev,double source, int N){
    //Apply Source
    auto tmp_idx = dev.d_Ein[0] + dev.d_Ein[1] * N;
    dev.d_V1[tmp_idx] = dev.d_V1[tmp_idx] + source;
    dev.d_V2[tmp_idx] = dev.d_V2[tmp_idx] - source;
    dev.d_V3[tmp_idx] = dev.d_V3[tmp_idx] - source;
    dev.d_V4[tmp_idx] = dev.d_V4[tmp_idx] + source;
}

/**
 *
 * @param dev
 * @param N
 * @param source
 */
__global__ void tlmScatter(dev_data dev, int N){

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    auto index = idx + idy * N;

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

/**
 *
 * @param dev
 * @param N
 * @param n
 */
__global__ void tlmConnect(dev_data dev, int N)
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

    //Apply Boundaries
    double rXmin = dev.coeff[1];
    double rXmax = dev.coeff[2];
    double rYmin = dev.coeff[3];
    double rYmax = dev.coeff[4];

    if (idy == N-1*N && idx < N*N){
        dev.d_V3[idx + (N - 1)*N] = rYmax * dev.d_V3[idx + (N - 1)*N];
        dev.d_V1[idx] = rYmin * dev.d_V1[idx];
    }

    if (idx == N-1 && idy < N*N) {
        dev.d_V4[(N - 1) + idy*N] = rXmax * dev.d_V4[(N - 1) + idy*N];
        dev.d_V2[idy*N] = rXmin * dev.d_V2[idy*N];
    }
}

/**
 *
 * @param dev
 * @param N
 * @param n
 */
__global__ void tlmApplyProbe(dev_data dev, int n, int N){
    auto tmp_idx = dev.d_Eout[0] + dev.d_Eout[1] * N;
    dev.out[n] = dev.d_V2[tmp_idx] + dev.d_V4[tmp_idx];
}

int main()
{
    //Specify Simulation Meta Parameters
    int NX = 100;                           // dim one of nodes
    int NY = 100;                           // dim 2 of nodes
    int NT = 8192;                          // number of time steps
    double dl = 1;                          // set node line segment length in metres
    double dt = dl / (sqrt(2.) * c);     // set time step duration


    //2D mesh variables
    double** V1 = declare_array2D(NX, NY);
    double** V2 = declare_array2D(NX, NY);
    double** V3 = declare_array2D(NX, NY);
    double** V4 = declare_array2D(NX, NY);
    double v_output[NT];
    for (int n = 0; n < NT; n++){
        v_output[n] = 0;
    }

    //boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;



    // specify mesh simulation parameters
    double Z = eta0 / sqrt(2.);
    double width = 20 * dt * sqrt(2.);
    double delay = 100 * dt * sqrt(2.);
    int Ein[] = { 10,10 };
    int Eout[] = { 15,15 };

    //Group Coefficients
    double coeff[] = {Z, rXmin, rXmax, rYmin, rYmax};

    //device arrays
    double* dev_V1;
    double* dev_V2;
    double* dev_V3;
    double* dev_V4;
    double* dev_coeff;
    double* dev_output;
    int* dev_Ein;
    int* dev_Eout;


    //allocate memory on device
    auto sz = NX * NY * sizeof(double);
    cudaMalloc((void**)&dev_V1, sz);
    cudaMalloc((void**)&dev_V2, sz);
    cudaMalloc((void**)&dev_V3, sz);
    cudaMalloc((void**)&dev_V4, sz);
    cudaMalloc((void**)&dev_coeff, sizeof(double)*6);
    cudaMalloc((void**)&dev_output, sizeof(double)*NT);
    cudaMalloc((void**)&dev_Ein, sizeof(int)*2);
    cudaMalloc((void**)&dev_Eout, sizeof(int)*2);



    //copy memory areas from host to device
    cudaMemcpy(dev_V1, V1, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V2, V2, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V3, V3, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V4, V4, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_coeff, coeff, sizeof(double)*6, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Ein, Ein, sizeof(int)*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Eout, Eout, sizeof(int)*2, cudaMemcpyHostToDevice);




    //Group Device Variables to simplify Kernel Calls
    dev_data dev_Data{dev_V1, dev_V2, dev_V3, dev_V4, dev_coeff, dev_output, dev_Ein, dev_Eout};
    //Determine Kernel Size
    dim3 dimBlock(10,10);
    dim3 dimGrid(ceil(NX/dimBlock.x),ceil(NY/dimBlock.y));

    //Start Timer
    auto t1 = std::chrono::high_resolution_clock::now();
    // Start of TLM algorithm
    //
    // loop over total time NT in steps of dt
    for (int n = 0; n < NT; n++)
    {
        //Calculate V Source for this delta
        double source = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
        //Apply the newly calculated Source
        tlmApplySource  <<<1, 1>>> (dev_Data, source, NX);
        //Apply Scatter Algorithm
        tlmScatter      <<<dimGrid, dimBlock>>> (dev_Data, NX);
        //Apply Connect Algorithm (Including Boundaries)
        tlmConnect      <<<dimGrid, dimBlock>>> (dev_Data, NX);
        //Get the Output from the mesh
        tlmApplyProbe   <<<1, 1>>> (dev_Data, n, NX);

    }
    //Get Result from Device
    cudaMemcpy(v_output, dev_output, sizeof(double)*NT, cudaMemcpyDeviceToHost);
    //Save output to file
    for (int n = 0; n < NT; n++){
        output << n * dt << "  " <<  v_output[n] << endl;
    }
    //End Timer
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2-t1;
    // End of TLM algorithm
    //Close output file
    output.close();
    //Signify finished
    cout << "Done";
    //Calculate time / Clocks
    std::cout << "\nExecuted in:   " << (diff.count()) << "s \n";
    cin.get();


    // free memory allocated on the GPU
    cudaFree(dev_V1);
    cudaFree(dev_V2);
    cudaFree(dev_V3);
    cudaFree(dev_V4);
    cudaFree(dev_output);
    cudaFree(dev_coeff);
    cudaFree(dev_Ein);
    cudaFree(dev_Eout);

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