#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

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

    // Improved Parallel Gaussian Elimination
    // Main computation loop for the Gaussian elimination
    for (i = 0; i < n - 1; i++)
    {
        int local_row = i / size;
        double local_pivot = 0.0;
        int pivot_owner = i % size; // Process that owns the pivot row
        int pivot_row = i;          // Initialize pivot_row to i, which is the starting assumption

        // Each process finds its local maximum if it owns the current row
        if (rank == pivot_owner)
        {
            for (j = i; j < n; j++)
            {
                if (fabs(local_matrix[local_row * n + j]) > local_pivot)
                {
                    local_pivot = fabs(local_matrix[local_row * n + j]);
                    pivot_row = j; // Local index of the pivot
                }
            }
        }

        // Define a structure to hold both the pivot value and its owner for reduction
        struct
        {
            double value;
            int rank;
        } local_pivot_struct = {local_pivot, rank}, global_pivot_struct;

        // Reduce to find the global maximum pivot and its owner
        MPI_Allreduce(&local_pivot_struct, &global_pivot_struct, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // Allocate buffer for the pivot row
        double *pivot_row_buffer = (double *)malloc(n * sizeof(double));

        // The owning process broadcasts the pivot row
        if (rank == global_pivot_struct.rank)
        {
            memcpy(pivot_row_buffer, local_matrix + local_row * n, n * sizeof(double));
        }

        MPI_Bcast(pivot_row_buffer, n, MPI_DOUBLE, global_pivot_struct.rank, MPI_COMM_WORLD);

        // Update local matrix rows based on the received pivot row
        for (j = 0; j < local_columns; j++)
        {
            if (rank * local_columns + j != pivot_row)
            {
                double factor = local_matrix[j * n + i] / pivot_row_buffer[i];
                for (k = 0; k < n; k++)
                {
                    local_matrix[j * n + k] -= factor * pivot_row_buffer[k];
                }
            }
        }

        free(pivot_row_buffer);
    }

    // Gather all parts of the matrix at the root process
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    // Calculate displacements and counts for gathering
    int sum = 0;
    for (i = 0; i < size; i++)
    {
        sendcounts[i] = (n + size - i - 1) / size * n; // Elements handled by each process
        displs[i] = sum;
        sum += sendcounts[i];
    }

    MPI_Gatherv(local_matrix, local_columns * n, MPI_DOUBLE, d0, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);



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
