/************************************************************************************
* FILE: gepp_mpi.c
* DESCRIPTION:
* MPI program for Gaussian elimination with partial pivoting using
* column block cyclic partitioning
* AUTHOR: Bing Bing Zhou, modified by AI
* LAST REVISED: 01/06/2024
*************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

void print_matrix(double** T, int rows, int cols);
void sequential_ge(double** a, int n);

int main(int argc, char* argv[])
{
    double* a0; // auxiliary 1D for 2D matrix a
    double** a; // 2D matrix for sequential computation

    int n; // input size
    int b; // block size
    int rank, size;
    int i, j, k, l;
    int indk;
    double c, amax;
    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;
    double* seq_result; // Declare seq_result here

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc == 3)
    {
        n = atoi(argv[1]);
        b = atoi(argv[2]);
        if (rank == 0)
        {
            printf("The matrix size:  %d * %d \n", n, n);
            printf("The block size: %d \n", b);
        }
    }
    else
    {
        if (rank == 0)
        {
            printf("Usage: %s n b\n\n"
                   " n: the matrix size\n"
                   " b: the block size\n\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Allocate contiguous memory for 2D matrices
    if (rank == 0)
    {
        printf("Creating and initializing matrices...\n\n");
        a0 = (double*)malloc(n*n*sizeof(double));
        a = (double**)malloc(n*sizeof(double*));
        for (i = 0; i < n; i++)
        {
            a[i] = a0 + i*n;
        }

        srand(time(0));
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                a[i][j] = (double)rand() / RAND_MAX;

        // Perform sequential Gaussian elimination
        printf("Starting sequential computation...\n\n");
        sequential_ge(a, n);

        // Save sequential results for comparison
        seq_result = (double*)malloc(n*n*sizeof(double));
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                seq_result[i*n + j] = a[i][j];
    }

    // Broadcast matrix size and block size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local blocks
    int local_cols = (n / b + (rank < n % b ? 1 : 0)) * b;
    double* local_a = (double*)malloc(n * local_cols * sizeof(double));
    double* recv_buffer = (double*)malloc(n * b * sizeof(double));

    // Scatter matrix columns to all processes
    if (rank == 0)
    {
        int* sendcounts = (int*)malloc(size * sizeof(int));
        int* displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (i = 0; i < size; i++)
        {
            sendcounts[i] = (n / size + (i < n % size ? 1 : 0)) * n;
            displs[i] = offset;
            offset += sendcounts[i];
        }

        MPI_Scatterv(a0, sendcounts, displs, MPI_DOUBLE, local_a, n * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        free(sendcounts);
        free(displs);
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, local_a, n * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
        gettimeofday(&start_time, 0);

    // Parallel Gaussian elimination with partial pivoting
    for (i = 0; i < n - 1; i++)
    {
        int owner = (i / b) % size;
        if (rank == owner)
        {
            // Find and broadcast pivot
            amax = local_a[(i % b) * local_cols + i];
            indk = i;
            for (k = i + 1; k < n; k++)
            {
                if (fabs(local_a[(i % b) * local_cols + k]) > fabs(amax))
                {
                    amax = local_a[(i % b) * local_cols + k];
                    indk = k;
                }
            }

            // Broadcast pivot index and value
            MPI_Bcast(&indk, 1, MPI_INT, owner, MPI_COMM_WORLD);
            MPI_Bcast(&amax, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            // Swap rows if needed
            if (indk != i)
            {
                for (j = 0; j < n; j++)
                {
                    c = local_a[(i % b) * local_cols + j];
                    local_a[(i % b) * local_cols + j] = local_a[(indk % b) * local_cols + j];
                    local_a[(indk % b) * local_cols + j] = c;
                }
            }
        }
        else
        {
            // Receive pivot index and value
            MPI_Bcast(&indk, 1, MPI_INT, owner, MPI_COMM_WORLD);
            MPI_Bcast(&amax, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        }

        // Perform elimination
        for (k = i + 1; k < n; k++)
        {
            if (rank == owner)
            {
                c = local_a[(i % b) * local_cols + k] / amax;
                local_a[(i % b) * local_cols + k] = c;
            }
            MPI_Bcast(&c, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            for (j = i + 1; j < n; j++)
            {
                local_a[(i % b) * local_cols + j] -= c * local_a[(i % b) * local_cols + j];
            }
        }
    }

    if (rank == 0)
    {
        gettimeofday(&end_time, 0);

        // Print the running time
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;
        elapsed = seconds + 1e-6 * microseconds;
        printf("parallel calculation time: %f\n\n", elapsed);
    }

    // Gather results back to rank 0 for verification
    if (rank == 0)
    {
        int* recvcounts = (int*)malloc(size * sizeof(int));
        int* displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (i = 0; i < size; i++)
        {
            recvcounts[i] = (n / size + (i < n % size ? 1 : 0)) * n;
            displs[i] = offset;
            offset += recvcounts[i];
        }

        MPI_Gatherv(local_a, n * local_cols, MPI_DOUBLE, a0, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        free(recvcounts);
        free(displs);
    }
    else
    {
        MPI_Gatherv(local_a, n * local_cols, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        // Compare parallel results with sequential results
        int correct = 1;
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                if (fabs(seq_result[i*n + j] - a0[i*n + j]) > 1e-6)
                {
                    correct = 0;
                    break;
                }
            }
        }

        if (correct)
        {
            printf("Parallel results match sequential results.\n");
        }
        else
        {
            printf("Parallel results do not match sequential results.\n");
        }

        // Free memory
        free(seq_result);
        free(a0);
        free(a);
    }

    // Free memory
    free(local_a);
    free(recv_buffer);

    MPI_Finalize();
    return 0;
}

void print_matrix(double** T, int rows, int cols)
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

void sequential_ge(double** a, int n)
{
    int i, j, k;
    int indk;
    double c, amax;

    for (i = 0; i < n - 1; i++)
    {
        // Find and record k where |a(k,i)| = max|a(j,i)|
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

        // Exit with a warning that a is singular
        if (amax == 0)
        {
            printf("Matrix is singular!\n");
            exit(1);
        }
        else if (indk != i) // Swap row i and row k
        {
            for (j = 0; j < n; j++)
            {
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
            }
        }

        // Store multiplier in place of A(k,i)
        for (k = i + 1; k < n; k++)
        {
            a[k][i] = a[k][i] / a[i][i];
        }

        // Subtract multiple of row a(i,:) to zero out a(j,i)
        for (k = i + 1; k < n; k++)
        {
            c = a[k][i];
            for (j = i + 1; j < n; j++)
            {
                a[k][j] -= c * a[i][j];
            }
        }
    }
}
