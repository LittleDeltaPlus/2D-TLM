//
// Created by deltaPlus on 04/01/2021.
//
#include<cstdio>
#include <chrono>
#include<iostream>
#include<fstream>
#include <ctime>
#include <cmath>
#define c 299792458
#define mu0 M_PI*4e-7
#define eta0 c*mu0

double** declare_array2D(int, int);

using namespace std;

int main() {
    std::clock_t start = std::clock();
    int NX = 100;
    int NY = 100;
    int NT = 8192;
    double dl = 1;
    double dt = dl / (sqrt(2.) * c);

    //2D mesh variables
    double I = 0, tempV = 0, E0 = 0, V = 0;
    double** V1 = declare_array2D(NX, NY);
    double** V2 = declare_array2D(NX, NY);
    double** V3 = declare_array2D(NX, NY);
    double** V4 = declare_array2D(NX, NY);

    double Z = eta0 / sqrt(2.);

    //boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    //input / output
    double width = 20 * dt * sqrt(2.);
    double delay = 100 * dt * sqrt(2.);
    int Ein[] = { 10,10 };
    int Eout[] = { 15,15 };

    ofstream output("output.out");

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < NT; n++) {
        auto lap_start = std::chrono::high_resolution_clock::now();
        //source
        E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
        V1[Ein[0]][Ein[1]] = V1[Ein[0]][Ein[1]] + E0;
        V2[Ein[0]][Ein[1]] = V2[Ein[0]][Ein[1]] - E0;
        V3[Ein[0]][Ein[1]] = V3[Ein[0]][Ein[1]] - E0;
        V4[Ein[0]][Ein[1]] = V4[Ein[0]][Ein[1]] + E0;

        //scatter
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                I = (2 * V1[x][y] + 2 * V4[x][y] - 2 * V2[x][y] - 2 * V3[x][y]) / (4 * Z);

                V = 2 * V1[x][y] - I * Z;         //port1
                V1[x][y] = V - V1[x][y];
                V = 2 * V2[x][y] + I * Z;         //port2
                V2[x][y] = V - V2[x][y];
                V = 2 * V3[x][y] + I * Z;         //port3
                V3[x][y] = V - V3[x][y];
                V = 2 * V4[x][y] - I * Z;         //port4
                V4[x][y] = V - V4[x][y];
            }
        }

        //connect
        for (int x = 1; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                tempV = V2[x][y];
                V2[x][y] = V4[x - 1][y];
                V4[x - 1][y] = tempV;
            }
        }
        for (int x = 0; x < NX; x++) {
            for (int y = 1; y < NY; y++) {
                tempV = V1[x][y];
                V1[x][y] = V3[x][y - 1];
                V3[x][y - 1] = tempV;
            }
        }

        //boundary
        for (int x = 0; x < NX; x++) {
            V3[x][NY - 1] = rYmax * V3[x][NY - 1];
            V1[x][0] = rYmin * V1[x][0];
        }
        for (int y = 0; y < NY; y++) {
            V4[NX - 1][y] = rXmax * V4[NX - 1][y];
            V2[0][y] = rXmin * V2[0][y];
        }
        auto lap_end = std::chrono::high_resolution_clock::now();
        output << n * dt << "  " << V2[Eout[0]][Eout[1]] + V4[Eout[0]][Eout[1]] << endl;
        if (n % 100 == 0){
            std::chrono::duration<double> diff = lap_end-lap_start;
            cout << n <<", "<< diff.count() << endl;
        }

    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2-t1;
    output.close();
    cout << "Done";
    std::cout << '\n' << "Executed in: " <<(diff.count()) << "s \n";
    cin.get();


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
