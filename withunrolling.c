#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>


void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);

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
    int block_size = 8;        // Column block size
    int loopUnrollFactor = 4; // Loop unrolling factor
    int rank, size;           // MPI rank and size

    double c;
    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;

    MPI_Init(&agrc, &agrv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (rank == 0)
    {
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

        printf("Starting sequential computation with loop unrolling and blocking...\n\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /***MPI with loop unrolling and blocking***/
    printf("Rank %d starting parallel computation...\n", rank);
    gettimeofday(&start_time, 0);

    // Define block cyclic distribution
    // Calculate number of columns this process will handle
    int num_local_columns = (n / block_size) * block_size / size + (rank < (n / block_size) % size ? block_size : 0);
    double *local_columns = malloc(num_local_columns * n * sizeof(double));

    // Create a type for one block column
    MPI_Datatype block_column_type;
    MPI_Type_vector(n, block_size, n, MPI_DOUBLE, &block_column_type);
    MPI_Type_commit(&block_column_type);

    // Scatter the blocks using the created type
    MPI_Scatter(d0, num_local_columns / block_size, block_column_type, local_columns, num_local_columns / block_size, block_column_type, 0, MPI_COMM_WORLD);


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

        n0 = (n - (i + 1)) / 4 * 4 + i + 1;

        for (k = i + 1; k < n0; k += 4)
        {
            for (int j = i + 1; j < n0; j += 4)
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
            // Handle remaining columns
            for (int j = n0; j < n; j++)
            {
                double dj = d[i][j];
                for (int m = 0; m < 4; m++)
                {
                    d[k + m][j] -= d[k + m][i] * dj;
                }
            }
        }

        for (k = n0; k < n; k++)
        {
            c = d[k][i];
            for (j = i + 1; j < n; j++)
                d[k][j] -= c * d[i][j];
        }
    }

    gettimeofday(&end_time, 0);

    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;

    MPI_Gather(local_columns, num_local_columns / block_size, block_column_type, d0, num_local_columns / block_size, block_column_type, 0, MPI_COMM_WORLD);


    if (rank == 0)
    {
        printf("MPI with loop unrolling time: %f\n\n", elapsed);
        printf("Starting comparison...\n\n");
        int cnt = test(a, d, n);
        if (cnt == 0)
        {
            printf("Done. There are no differences!\n");
        }
        else
        {
            printf("Results are incorrect! The number of different elements is %d\n", cnt);
        }
    }
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
