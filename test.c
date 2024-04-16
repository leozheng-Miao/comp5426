/******************************************************************************
* FILE: gepp_3.c
* DESCRIPTION:
* The C program for Gaussian elimination with partial pivoting
* Try to use loop unrolling to improve the performance - third attempt
* Unroll both j and k loops in rank 1 updating for trailing submatrix 
*   with unrolling factor = 4
* The performance is better than the first two attempts as data loaded into 
*   registers can be used multiple times before being replaced
* We can see a big performance improvement when compiling the program
* without using optimization options
* However, if we use "gcc -O3", this loop unrolling program only chieved 
* around 10% performance improvement
* Therefore, the program needs a further revision to enhance the performance
* AUTHOR: Bing Bing Zhou
* LAST REVISED: 1/06/2023
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);
void update_submatrix(double **d, int i, int k, int n, int n0);
void process_blocks(double **d, int n);

int main(int agrc, char *agrv[])
{
    double *a0; //auxiliary 1D for 2D matrix a
    double **a; //2D matrix for sequential computation
    double *d0; //auxiliary 1D for 2D matrix d
    double **d; //2D matrix, same initial data as a for computation with loop unrolling
    int n;      //input size
    int n0;
    int i, j, k;
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

    /***sequential computation with loop unrolling***/
    gettimeofday(&start_time, 0);
    process_blocks(d, n);

    // for (i = 0; i < n - 1; i++)
    // {
    //     amax = d[i][i];
    //     indk = i;
    //     for (k = i + 1; k < n; k++)
    //         if (fabs(d[k][i]) > fabs(amax))
    //         {
    //             amax = d[k][i];
    //             indk = k;
    //         }

    //     if (amax == 0.0)
    //     {
    //         printf("the matrix is singular\n");
    //         exit(1);
    //     }
    //     else if (indk != i) //swap row i and row k
    //     {
    //         for (j = 0; j < n; j++)
    //         {
    //             c = d[i][j];
    //             d[i][j] = d[indk][j];
    //             d[indk][j] = c;
    //         }
    //     }

    //     for (k = i + 1; k < n; k++)
    //         d[k][i] = d[k][i] / d[i][i];

    //     n0 = (n - (i + 1)) / 4 * 4 + i + 1;

    //     for (k = i + 1; k < n0; k += 4)
    //     {
    //         for (int j = i + 1; j < n0; j += 4)
    //         {
    //             double di[] = {d[k][i], d[k + 1][i], d[k + 2][i], d[k + 3][i]};
    //             double dj[] = {d[i][j], d[i][j + 1], d[i][j + 2], d[i][j + 3]};
    //             for (int m = 0; m < 4; m++)
    //             {
    //                 for (int n = 0; n < 4; n++)
    //                 {
    //                     d[k + m][j + n] -= di[m] * dj[n];
    //                 }
    //             }
    //         }
    //         // Handle remaining columns
    //         for (int j = n0; j < n; j++)
    //         {
    //             double dj = d[i][j];
    //             for (int m = 0; m < 4; m++)
    //             {
    //                 d[k + m][j] -= d[k + m][i] * dj;
    //             }
    //         }
    //     }

    //     for (k = n0; k < n; k++)
    //     {
    //         c = d[k][i];
    //         for (j = i + 1; j < n; j++)
    //             d[k][j] -= c * d[i][j];
    //     }
    // }

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

void process_blocks(double **d, int n)
{
    int block_size = 4;
    for (int i = 0; i < n; i += block_size)
    {
        for (int j = 0; j < n; j += block_size)
        {
            for (int k = i; k < i + block_size; k++)
            {
                update_submatrix(d, i, k, n, j + block_size);
            }
        }
    }
}

void update_submatrix(double **d, int i, int k, int n, int n0)
{
    for (int j = i; j < n0; j += 4)
    {
        double di[] = {d[k][i], d[k + 1][i], d[k + 2][i], d[k + 3][i]};
        double dj[] = {d[i][j], d[i][j + 1], d[i][j + 2], d[i][j + 3]};
        for (int m = 0; m < 4; m++)
        {
            for (int n = 0; n < 4; n++)
            {
                d[k + m][j + n] -= di[m] * dj[n];
            }
        }
    }
    for (int j = n0; j < n; j++)
    {
        double dj = d[i][j];
        for (int m = 0; m < 4; m++)
        {
            d[k + m][j] -= d[k + m][i] * dj;
        }
    }
}