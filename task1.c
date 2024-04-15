#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

void print_matrix(double** T, int rows, int cols);

int main(int argc, char* argv[])
{
    //copy element for self-check
    double* a0, *a0_copy;
    double** a, **a_copy;

    // Block size set to 4  
    int n, b = 4; 
    int i, j, k, ii, jj, kk;
    int indk;
    double c, amax;

    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;

    if (argc == 2)
    {
        n = atoi(argv[1]);
        printf("The matrix size: %d * %d \n", n, n);
    }
    else
    {
        printf("Usage: %s n\n\n" 
                " n: the matrix size\n\n", argv[0]);
        return 1;
    }

    // Memory allocation and initialization for gepp_0 and revised version
    a0 = (double*)malloc(n*n*sizeof(double));
    a = (double**)malloc(n*sizeof(double*));

    a0_copy = (double*)malloc(n*n*sizeof(double));
    a_copy = (double**)malloc(n*sizeof(double*));

    for (i=0; i<n; i++)
    {
        a[i] = a0 + i*n;
        a_copy[i] = a0_copy + i*n;
    }

    srand(time(0));
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            a[i][j] = (double)rand()/RAND_MAX;
            //copy the matrix for validation
            a_copy[i][j] = a[i][j];

        }
    }
        
    printf("Starting sequential computation with block and unrolling...\n\n");
    gettimeofday(&start_time, 0);

    // Applying blocking and loop unrolling
    for (ii = 0; ii < n; ii += b) {
        for (jj = 0; jj < n; jj += b) {
            for (kk = 0; kk < n; kk += b) {
                for (i = ii; i < ii+b && i < n; i++) {
                    for (j = jj; j < jj+b && j < n; j++) {
                        for (k = kk; k < kk+b && k < n; k+=4) { // Unrolling k-loop
                            a[i][j] -= a[i][k] * a[k][j];
                            a[i][j] -= a[i][k+1] * a[k+1][j];
                            a[i][j] -= a[i][k+2] * a[k+2][j];
                            a[i][j] -= a[i][k+3] * a[k+3][j];
                        }
                    }
                }
            }
        }
    }

    gettimeofday(&end_time, 0);
    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
    printf("Enhanced sequential calculation time: %f\n\n", elapsed);
    print_matrix(a, n, n);

    //Use copy element to cluculate original computation without optimizations
    printf("Starting original computation...\n\n");
    gettimeofday(&start_time, 0);
    for(i = 0; i < n - 1; i++) {
        for( j = i + 1; j < n; j++) {
            a_copy[j][i] = a_copy[j][i] / a_copy[i][i];
            for(k = i + 1; k < n; k++) {
                a_copy[j][k] = a_copy[j][i] * a_copy[i][k];
            }
        }
    }
    gettimeofday(&end_time, 0);
    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
    printf("Original sequential calculation time: %f\n\n", elapsed);

    //Comparison for correctness
    int error_count = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (fabs(a[i][j] - a_copy[i][j]) > 1e-10) {
                error_count++;
            }
        }
    }

    print_matrix(a_copy, n, n);


    if (error_count == 0)
        printf("Results are correct. No discrepancies found.\n");
    else
        printf("Results are incorrect! Discrepancies found: %d\n", error_count);



    free(a0);
    free(a);
    free(a0_copy);
    free(a_copy);
    return 0;
}

void print_matrix(double** T, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", T[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

