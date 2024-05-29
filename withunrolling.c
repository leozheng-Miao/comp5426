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
    int n, b = 8;    // Matrix size and block size
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

    if (argc != 2)
    {
        if (rank == 0)
        {
            printf("Usage: %s n b\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    n = atoi(argv[1]);

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
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /**** MPI without rool unrolling *****/

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

    printf("Starting mpi with loop unrolling calculation\n\n");
    gettimeofday(&start_time, NULL);
    // Parallel Gaussian elimination
    for (i = 0; i < n - 1; i++)
    {
        double pivot = 0.0;
        int pivot_row = i;
        double *row_buffer = (double *)malloc(n * sizeof(double));
        double local_max = 0.0; // To find local maximum for pivot

        // Find local maximum for the pivot
        if (rank == i % size)
        {
            int local_row = i / size;
            for (j = i; j < n; j++)
            {
                if (fabs(local_matrix[local_row * n + j]) > local_max)
                {
                    local_max = fabs(local_matrix[local_row * n + j]);
                    pivot_row = j;
                }
            }
            pivot = local_matrix[local_row * n + pivot_row];                      // Local pivot found
            memcpy(row_buffer, local_matrix + local_row * n, n * sizeof(double)); // Copy the row
        }

        // Broadcast the pivot information and the entire pivot row
        MPI_Allreduce(MPI_IN_PLACE, &pivot_row, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Bcast(&pivot, 1, MPI_DOUBLE, pivot_row % size, MPI_COMM_WORLD);
        MPI_Bcast(row_buffer, n, MPI_DOUBLE, pivot_row % size, MPI_COMM_WORLD);

        // Update local matrix using the received pivot row and loop unrolling
        for (j = 0; j < local_columns; j++)
        {
            int col_index = rank * local_columns + j;
            if (col_index != pivot_row)
            {
                double factor = local_matrix[j * n + i] / pivot;
                int k;
                for (k = i + 1; k <= n - 4; k += 4) // Ensures that we have at least four elements left
                {
                    local_matrix[j * n + k] -= factor * row_buffer[k];
                    local_matrix[j * n + k + 1] -= factor * row_buffer[k + 1];
                    local_matrix[j * n + k + 2] -= factor * row_buffer[k + 2];
                    local_matrix[j * n + k + 3] -= factor * row_buffer[k + 3];
                }
                for (; k < n; k++) // Handle remaining elements
                {
                    local_matrix[j * n + k] -= factor * row_buffer[k];
                }
            }
        }
        free(row_buffer);
    }

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int sum = 0; // This will calculate the displacement
    for (i = 0; i < size; i++)
    {
        sendcounts[i] = ((n + size - i - 1) / size) * n; // Calculate number of elements to send
        displs[i] = sum;
        sum += sendcounts[i]; // Update sum
    }

    // gather the results at root
    MPI_Gatherv(local_matrix, local_columns * n, MPI_DOUBLE, d0, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);

    MPI_Type_free(&column_type);
    free(local_matrix);

    gettimeofday(&end_time, NULL);

    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;

    if (rank == 0)
    {
        printf("Print MPI");
        print_matrix(d, n, n);

        printf("Print origin");
        print_matrix(a, n, n);

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
