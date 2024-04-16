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

    /***sequential computation with loop unrolling***/
    gettimeofday(&start_time, 0);
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

        // for (kk = i + 1; kk < n; kk += b)
        // {
        //     int Kmax = (kk + b < n) ? kk + b : n;

        //     for (jj = i + 1; jj < n; jj += b)
        //     {
        //         int Jmax = (jj + b < n) ? jj + b : n;

        //         for (k = kk; k < Kmax; k += 4)
        //         {
        //             di00 = d[k][i];
        //             di10 = d[k + 1][i];
        //             di20 = d[k + 2][i];
        //             di30 = d[k + 3][i];

        //             for (j = jj; j < Jmax; j += 4)
        //             {
        //                 dj00 = d[i][j];
        //                 dj01 = d[i][j + 1];
        //                 dj02 = d[i][j + 2];
        //                 dj03 = d[i][j + 3];

        //                 d[k][j] -= di00 * dj00;
        //                 d[k][j + 1] -= di00 * dj01;
        //                 d[k][j + 2] -= di00 * dj02;
        //                 d[k][j + 3] -= di00 * dj03;
        //                 d[k + 1][j] -= di10 * dj00;
        //                 d[k + 1][j + 1] -= di10 * dj01;
        //                 d[k + 1][j + 2] -= di10 * dj02;
        //                 d[k + 1][j + 3] -= di10 * dj03;
        //                 d[k + 2][j] -= di20 * dj00;
        //                 d[k + 2][j + 1] -= di20 * dj01;
        //                 d[k + 2][j + 2] -= di20 * dj02;
        //                 d[k + 2][j + 3] -= di20 * dj03;
        //                 d[k + 3][j] -= di30 * dj00;
        //                 d[k + 3][j + 1] -= di30 * dj01;
        //                 d[k + 3][j + 2] -= di30 * dj02;
        //                 d[k + 3][j + 3] -= di30 * dj03;
        //             }
        //         }
        //     }
        // }

        // Apply blocking technique
        // for (int bi = i + 1; bi < n; bi += b)
        // {
        //     int blockEndI = min(bi + b, n);
        //     for (int bj = i + 1; bj < n; bj += b)
        //     {
        //         int blockEndJ = min(bj + b, n);
        //         // Unrolling the j loop inside the block
        //         for (k = bi; k < blockEndI; k++)
        //         {
        //             register double dki = d[k][i];
        //             for (j = bj; j < blockEndJ; j += 4)
        //             {
        //                 if (n - j >= 4)
        //                 {
        //                     // Loop unrolling for the subtraction step
        //                     d[k][j] -= dki * d[i][j];
        //                     d[k][j + 1] -= dki * d[i][j + 1];
        //                     d[k][j + 2] -= dki * d[i][j + 2];
        //                     d[k][j + 3] -= dki * d[i][j + 3];
        //                 }
        //                 else
        //                 {
        //                     // Handling the case where the block size is not a multiple of 4
        //                     for (int remain = j; remain < blockEndJ; remain++)
        //                     {
        //                         d[k][remain] -= dki * d[i][remain];
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // for (int ii = i + 1; ii < n; ii += b)
        // {
        //     for (int jj = i + 1; jj < n; jj += b)
        //     {
        //         int i_max = min(ii + b, n);
        //         int j_max = min(jj + b, n);

        //         // Process the block
        //         for (int ik = ii; ik < i_max; ik++)
        //         {
        //             for (int jk = jj; jk < j_max; jk += 4)
        //             {
        //                 // Make sure we have enough space to unroll the loop
        //                 if (jk + 3 < n)
        //                 {
        //                     d[ik][jk] -= d[ik][i] * d[i][jk];
        //                     d[ik][jk + 1] -= d[ik][i] * d[i][jk + 1];
        //                     d[ik][jk + 2] -= d[ik][i] * d[i][jk + 2];
        //                     d[ik][jk + 3] -= d[ik][i] * d[i][jk + 3];
        //                 }
        //                 else
        //                 {
        //                     // For the edge case where we can't unroll the loop completely
        //                     for (int jk_edge = jk; jk_edge < j_max; jk_edge++)
        //                     {
        //                         d[ik][jk_edge] -= d[ik][i] * d[i][jk_edge];
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
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
    int j, k, r;
    
    for (r = i + 1; r < n; r += b) {  // Loop over blocks of rows
        for (k = i + 1; k < n; ++k) {
            double Aki = A[k][i];  // Elimination factor
            for (j = r; j < min(r + b, n); j += 4) {  // Loop unrolling by 4
                A[k][j] -= Aki * A[i][j];
                if (j + 1 < n) A[k][j + 1] -= Aki * A[i][j + 1];
                if (j + 2 < n) A[k][j + 2] -= Aki * A[i][j + 2];
                if (j + 3 < n) A[k][j + 3] -= Aki * A[i][j + 3];
            }
        }
    }
}




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
