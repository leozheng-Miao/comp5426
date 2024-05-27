#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

void print_matrix(double** T, int rows, int cols);

int main(int argc, char* argv[]) {
    int rank, size;
    int n, b; // Matrix size and block size
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

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s n b\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    n = atoi(argv[1]);
    b = atoi(argv[2]);

    a0 = (double*) malloc(n * n * sizeof(double));
    a = (double**) malloc(n * sizeof(double*));
    d0 = (double*) malloc(n * n * sizeof(double));
    d = (double**) malloc(n * sizeof(double*));
    for (i = 0; i < n; i++) {
        a[i] = a0 + i * n;
        d[i] = d0 + i * n;
    }

    // Initialize matrix with random values
    srand(time(NULL) * rank); // Different seed per process
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            a[i][j] = d[i][j] = (double)rand() / RAND_MAX;
        }
    }

    
    printf("Starting sequential computation...\n\n"); 
    /**** Sequential computation *****/
    gettimeofday(&start_time, 0);
    for (i=0; i<n-1; i++)
    {
        //find and record k where |a(k,i)|=𝑚ax|a(j,i)|
        amax = a[i][i];
        indk = i;
        for (k=i+1; k<n; k++)
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
            for (j=0; j<n; j++)
            {
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
            }
        } 

        //store multiplier in place of A(k,i)
        for (k=i+1; k<n; k++)
        {
            a[k][i] = a[k][i]/a[i][i];
        }

        //subtract multiple of row a(i,:) to zero out a(j,i)
        for (k=i+1; k<n; k++)
        { 
            c = a[k][i]; 
            for (j=i+1; j<n; j++)
            {
                a[k][j] -= c*a[i][j];
            }
        }
    }
    gettimeofday(&end_time, 0);
 
    //print the running time
    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
    printf("sequential calculation time: %f\n\n",elapsed); 

    /**** MPI without rool unrolling *****/

    gettimeofday(&start_time, NULL);

    //derived datatype for column block cyclis partitioning
    MPI_Datatype column_type;
    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    //Distribute columns using block cyclic partitioning 
    for (j = 0; j < n; j++) {
        int target_process = (j / b) % size;
        if (rank == 0)
        {
            if (target_process != 0)
            {
                MPI_Send(d0 + j * n, 1, column_type, target_process, 0, MPI_COMM_WORLD);
            }
            
        } else if (rank == target_process) {
            MPI_Recv(d0 + j * n, 1, column_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
    }

    // Perform parallel Gaussian elimination
    for (i = 0; i < n - 1; i++) {
        // Broadcast current row
        if (rank == (i / b) % size) {
            MPI_Bcast(d[i], n, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        }

        for (k = i + 1; k < n; k++) {
            if (rank == (k / b) % size) {
                d[k][i] = d[k][i] / d[i][i];
                for (j = i + 1; j < n; j++) {
                    d[k][j] -= d[k][i] * d[i][j];
                }
            }
        }
    }



    gettimeofday(&end_time, NULL);

    seconds = end_time.tv_sec - start_time.tv_sec;
    microseconds = end_time.tv_usec - start_time.tv_usec;
    elapsed = seconds + 1e-6 * microseconds;
    printf("sequential calculation with loop unrolling and blocking time: %f\n\n", elapsed);

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

void print_matrix(double** T, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
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
