#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);

int main(int argc, char *argv[])
{
    int rank, size;
    int n, b;        // Matrix size and block size
    double *a0, *d0; // Auxiliary 1D for 2D matrix a
    double **a, **d; // 2D matrix for computation
    int i, j, k, indk;
    double c, amax;

    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;

    //Processes are organized as a one dimensional, or 1D array
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3)
    {
        if (rank == 0)
        {
            printf("Usage: %s n b\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    n = atoi(argv[1]);
    b = atoi(argv[2]);

    a0 = (double *)malloc(n * n * sizeof(double));
    a = (double **)malloc(n * sizeof(double *));
    d0 = (double *)malloc(n * n * sizeof(double));
    d = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        a[i] = a0 + i * n;
        d[i] = d0 + i * n;
    }

    // Initialize matrix with random values
    srand(time(NULL) * rank); // Different seed per process
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i][j] = d[i][j] = (double)rand() / RAND_MAX;
        }
    }

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

        //store multiplier in place of A(k,i)
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

    /**** MPI without rool unrolling *****/

    printf("Starting mpi without loop unrolling calculation\n\n");
    gettimeofday(&start_time, NULL);

    //derived datatype for column block cyclis partitioning
    int local_columns = (n + b - 1) / b;
    double *local_matrix = malloc(local_columns * n * sizeof(double));

    //Create derived datatype for a column
    MPI_Datatype column_type;
    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    for (j = 0; j < local_columns; j++)
    {
        int col = rank * local_columns + j;
        if (col < n)
        {
            for (i = 0; i < n; i++)
            {
                local_matrix[j * n + i] = a[i][col]; // Transpose block to local storage
            }
        }
    }

    // Parallel Gaussian elimination (simplified for example)
    for (i = 0; i < n - 1; i++)
    {
        // Broadcast pivot elements
        if (i % size == rank)
        {
            k = i / size;                   // Local index of pivot row
            amax = local_matrix[k * n + i]; // Pivot value
            indk = i;
        }

        MPI_Bcast(&amax, 1, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
        MPI_Bcast(&indk, 1, MPI_INT, i % size, MPI_COMM_WORLD);

        // Perform local computations on each block
        for (j = 0; j < local_columns; j++)
        {
            int col = rank * local_columns + j;
            if (col >= i + 1)
            {
                local_matrix[j * n + i] /= amax; // Update matrix
                for (k = i + 1; k < n; k++)
                {
                    local_matrix[j * n + k] -= local_matrix[j * n + i] * a[k][i];
                }
            }
        }
    }

    // Copy results back to 'd' for verification
    for (j = 0; j < local_columns; j++)
    {
        int col = rank * local_columns + j;
        if (col < n)
        {
            for (i = 0; i < n; i++)
            {
                d[i][col] = local_matrix[j * n + i];
            }
        }
    }

    MPI_Type_free(&column_type);
    free(local_matrix);

    gettimeofday(&end_time, NULL);

    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
    printf("mpi without loop unrolling time: %f\n\n", elapsed);

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
    MPI_Finalize();
    return 0;
}

void print_matrix(double **T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f ", T[i][j]);
        }
        printf("\n");
    }
    printf("\n");
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
