
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

#define BLOCK_SIZE 4


void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);

int main(int agrc, char *agrv[])
{
    double *a0; //auxiliary 1D for 2D matrix a
    double **a; //2D matrix for sequential computation
    double *d0; //auxiliary 1D for 2D matrix d
    double **d; //2D matrix, same initial data as a for computation with loop unrolling
    int n, T;   //input size
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

    if (agrc == 3)
    {
        n = atoi(agrv[1]);
        T = atoi(agrv[2]);

        printf("The matrix size:  %d * %d \n", n, n);
    }
    else
    {
        printf("Usage: %s n\n\n"
               " n: the matrix size\n",
               agrv[0]);
        return 1;
    }

    omp_set_num_threads(T);

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

    printf("Parallel computation......\n\n");

    gettimeofday(&start_time, 0);
    /*** Parallel computation ***/
    // #pragma omp parallel shared(d, n) private(i, j, k, amax, indk, c)
    //     {
    //         for (i = 0; i < n - 1; i++)
    //         {
    // #pragma omp for schedule(dynamic) nowait
    //             for (k = i + 1; k < n; k++)
    //             {
    //                 if (fabs(d[k][i]) > fabs(d[i][i]))
    //                 {
    // #pragma omp critical
    //                     {
    //                         if (fabs(d[k][i]) > fabs(d[i][i]))
    //                         { // Double-checked locking
    //                             d[i][i] = d[k][i];
    //                             indk = k;
    //                         }
    //                     }
    //                 }
    //             }

    // #pragma omp single
    //             {
    //                 if (d[i][i] == 0.0)
    //                 {
    //                     printf("Matrix is singular.\n");
    //                     exit(1);
    //                 }
    //                 else if (indk != i)
    //                 {
    //                     for (j = 0; j < n; j++)
    //                     {
    //                         c = d[i][j];
    //                         d[i][j] = d[indk][j];
    //                         d[indk][j] = c;
    //                     }
    //                 }
    //             }

    // #pragma omp for
    //             for (k = i + 1; k < n; k++)
    //             {
    //                 d[k][i] /= d[i][i];
    //                 for (j = i + 1; j < n; j++)
    //                 {
    //                     d[k][j] -= d[k][i] * d[i][j];
    //                 }
    //             }
    //         }
    //     }

    for (int bi = 0; bi < n; bi += BLOCK_SIZE)
    {
        int bimax = bi + BLOCK_SIZE < n ? bi + BLOCK_SIZE : n;
        for (i = bi; i < bimax - 1; i++)
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
            else if (indk != i)
            {
#pragma omp parallel for private(j, c) // Parallelize the row swapping
                for (j = 0; j < n; j++)
                {
                    c = d[i][j];
                    d[i][j] = d[indk][j];
                    d[indk][j] = c;
                }
            }

#pragma omp parallel for private(k, c) // Parallelize the division for the pivot row
            for (k = i + 1; k < n; k++)
                d[k][i] = d[k][i] / d[i][i];

            // Adjust the loop for blocking
            for (int bj = i + 1; bj < n; bj += BLOCK_SIZE)
            {
                int bjmax = bj + BLOCK_SIZE < n ? bj + BLOCK_SIZE : n;
#pragma omp parallel for private(k, j, c) collapse(2) // Parallelize the main updating matrix
                for (k = i + 1; k < n; k++)
                {
                    for (j = bj; j < bjmax; j++)
                    {
                        d[k][j] -= d[k][i] * d[i][j];
                    }
                }
            }
        }
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

    free(a0);
    free(a);
    free(d0);
    free(d);
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
