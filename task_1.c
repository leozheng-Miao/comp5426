#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);
int min(int a, int b);
void gepp_with_blocking_and_unrolling(double **d, int n, int i, int b);

int main(int agrc, char *agrv[])
{
    double *a0;   //auxiliary 1D for 2D matrix a
    double **a;   //2D matrix for sequential computation
    double *d0;   //auxiliary 1D for 2D matrix d
    double **d;   //2D matrix, same initial data as a for computation with loop unrolling
    int n, b = 4; //input size
    int n0;
    int i, j, k, kk, jj;
    int indk;
    double amax;
    register double di00, di10, di20, di30;
    register double dj00, dj01, dj02, dj03;
    double c;
    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;

    if (agrc == 2)
    {
        n = atoi(agrv[1]);
        printf("The matrix size:  %d * %d \n", n, n);
    }
    else
    {
        printf("Usage: %s n\n\n"
               " n: the matrix size\n",
               agrv[0]);
        return 1;
    }

    printf("Creating and initializing matrices...\n\n");
    /*** Allocate contiguous memory for 2D matrices ***/
    a0 = (double *)malloc(n * n * sizeof(double));
    a = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        a[i] = a0 + i * n;
    }
    d0 = (double *)malloc(n * n * sizeof(double));
    d = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        d[i] = d0 + i * n;
    }

    srand(time(0));
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i][j] = (double)rand() / RAND_MAX;
            d[i][j] = a[i][j];
        }
    }
    //    printf("matrix a: \n");
    //    print_matrix(a, n, n);
    //    printf("matrix d: \n");
    //    print_matrix(d, n, n);

    printf("Starting sequential computation...\n\n");
    /**** Sequential computation *****/
    gettimeofday(&start_time, 0);
    for (i = 0; i < n - 1; i++)
    {
        //find and record k where |a(k,i)|=𝑚ax|a(j,i)|
        amax = a[i][i];
        indk = i;
        for (k = i + 1; k < n; k++)
        {
            if (fabs(a[k][i]) > fabs(amax))
            {
                amax = a[k][i];
                indk = k;
            }
        }

        //exit with a warning that a is singular
        if (amax == 0)
        {
            printf("matrix is singular!\n");
            exit(1);
        }
        else if (indk != i) //swap row i and row k
        {
            for (j = 0; j < n; j++)
            {
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
            }
        }

        //store multiplier in place of A(j,i)
        for (k = i + 1; k < n; k++)
        {
            a[k][i] = a[k][i] / a[i][i];
        }

        //subtract multiple of row a(i,:) to zero out a(j,i)
        for (k = i + 1; k < n; k++)
        {
            c = a[k][i];
            for (j = i + 1; j < n; j++)
            {
                a[k][j] -= c * a[i][j];
            }
        }
    }
    gettimeofday(&end_time, 0);

    //print the running time
    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
    printf("sequential calculation time: %f\n\n", elapsed);

    printf("Starting sequential computation with loop unrolling...\n\n");

    /***sequential computation with loop unrolling and blocking***/
    gettimeofday(&start_time, 0);
    int block_size = 4;       // Define block size
    int unrolling_factor = 4; // Define unrolling factor

    for (i = 0; i < n - 1; i++)
    {
        amax = d[i][i];
        indk = i;
        for (k = i + 1; k < n; k++)
            if (fabs(d[k][i]) > fabs(amax))
            {
                amax = d[k][i];
                indk = k;
            }

        if (amax == 0.0)
        {
            printf("the matrix is singular\n");
            exit(1);
        }
        else if (indk != i) //swap row i and row k
        {
            for (j = 0; j < n; j++)
            {
                c = d[i][j];
                d[i][j] = d[indk][j];
                d[indk][j] = c;
            }
        }

        for (k = i + 1; k < n; k++)
            d[k][i] = d[k][i] / d[i][i];


        gepp_with_blocking_and_unrolling(d, n, i, b);




    }
    gettimeofday(&end_time, 0);

    //print the running time
    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
    printf("sequential calculation with loop unrolling time: %f\n\n", elapsed);

    printf("Starting comparison...\n\n");
    int cnt;
    cnt = test(a, d, n);
    if (cnt == 0)
        printf("Done. There are no differences!\n");
    else
        printf("Results are incorrect! The number of different elements is %d\n", cnt);
}

void print_matrix(double **T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f   ", T[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
    return;
}

int test(double **t1, double **t2, int rows)
{
    int i, j;
    int cnt;
    cnt = 0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            if ((t1[i][j] - t2[i][j]) * (t1[i][j] - t2[i][j]) > 1.0e-16)
            {
                cnt += 1;
            }
        }
    }

    return cnt;
}

int min(int a, int b)
{
    return a < b ? a : b;
}

void gepp_with_blocking_and_unrolling(double **A, int n, int i, int b) {
    int k, j, r;

    // Elimination step
    for (k = i + 1; k < n; ++k) {
        for (r = i + 1; r < n; r += b) {
            double Aki = A[k][i]; // Store this multiplication factor to avoid recomputing it
            int rMax = min(r + b, n);
            for (j = r; j < rMax; j += 4) { // Unroll by 4
                // Make sure to not exceed the matrix dimension
                A[k][j] -= Aki * A[i][j];
                if (j + 1 < rMax) A[k][j + 1] -= Aki * A[i][j + 1];
                if (j + 2 < rMax) A[k][j + 2] -= Aki * A[i][j + 2];
                if (j + 3 < rMax) A[k][j + 3] -= Aki * A[i][j + 3];
            }
            // Clean-up loop for remaining elements when n is not a multiple of 4
            for (; j < rMax; ++j) {
                A[k][j] -= Aki * A[i][j];
            }
        }
    }
}

// void gepp_with_blocking_and_unrolling(double **A, int n, int i, int b)
// {
//     // double Aii_inv = 1.0 / A[i][i]; // Inverse of the pivot element
//     int j, k;

//     // Perform the division outside of the loop to avoid division inside the loop
//     // for (k = i + 1; k < n; ++k) {
//     //     A[k][i] *= Aii_inv;
//     // }

//     // Loop tiling for cache optimization
//     for (k = i + 1; k < n; ++k)
//     {
//         double Aki = A[k][i]; // Store this multiplication factor to avoid recomputing it
//         // Main operation with loop unrolling
//         for (j = i + 1; j <= n - 4; j += 4)
//         {
//             A[k][j] -= Aki * A[i][j];
//             A[k][j + 1] -= Aki * A[i][j + 1];
//             A[k][j + 2] -= Aki * A[i][j + 2];
//             A[k][j + 3] -= Aki * A[i][j + 3];
//         }
//         // Tail case handling when n is not a multiple of 4
//         for (; j < n; ++j)
//         {
//             A[k][j] -= Aki * A[i][j];
//         }
//     }
// }

// void gepp_with_blocking_and_unrolling(double **d, int n, int i, int b)
// {
//     int k, j, kk, jj;
//     double c, di00, di10, di20, di30, dj00, dj01, dj02, dj03;
//     int n0 = (n - (i + 1)) / 4 * 4 + i + 1;

//     for (kk = i + 1; kk < n; kk += b)
//     {
//         int Kmax = (kk + b < n) ? kk + b : n;

//         for (jj = i + 1; jj < n; jj += b)
//         {
//             int Jmax = (jj + b < n) ? jj + b : n;

//             for (k = kk; k < Kmax; k += 4)
//             {
//                 di00 = d[k][i];
//                 di10 = d[k + 1][i];
//                 di20 = d[k + 2][i];
//                 di30 = d[k + 3][i];

//                 for (j = jj; j < Jmax; j += 4)
//                 {
//                     dj00 = d[i][j];
//                     dj01 = d[i][j + 1];
//                     dj02 = d[i][j + 2];
//                     dj03 = d[i][j + 3];

//                     d[k][j] -= di00 * dj00;
//                     d[k][j + 1] -= di00 * dj01;
//                     d[k][j + 2] -= di00 * dj02;
//                     d[k][j + 3] -= di00 * dj03;
//                     d[k + 1][j] -= di10 * dj00;
//                     d[k + 1][j + 1] -= di10 * dj01;
//                     d[k + 1][j + 2] -= di10 * dj02;
//                     d[k + 1][j + 3] -= di10 * dj03;
//                     d[k + 2][j] -= di20 * dj00;
//                     d[k + 2][j + 1] -= di20 * dj01;
//                     d[k + 2][j + 2] -= di20 * dj02;
//                     d[k + 2][j + 3] -= di20 * dj03;
//                     d[k + 3][j] -= di30 * dj00;
//                     d[k + 3][j + 1] -= di30 * dj01;
//                     d[k + 3][j + 2] -= di30 * dj02;
//                     d[k + 3][j + 3] -= di30 * dj03;
//                 }
//             }
//         }
//     }
// }
