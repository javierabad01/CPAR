// ABAD HERNÁNDEZ, JAVIER
// CASTRO GARCÍA, JAIME

//============================================================================
// Name:			KMEANS.c
// Compilacion:	gcc KMEANS.c -o KMEANS -lm
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <mpi.h> // Incluir encabezado de MPI

// Constantes
#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))








/*
Muestra el correspondiente errro en la lectura de fichero de datos
*/
void showFileError(int error, char *filename)
{
    printf("Error\n");
    switch (error)
    {
    case -1:
        fprintf(stderr, "\tEl fichero %s contiene demasiadas columnas.\n", filename);
        fprintf(stderr, "\tSe supero el tamano maximo de columna MAXLINE: %d.\n", MAXLINE);
        break;
    case -2:
        fprintf(stderr, "Error leyendo el fichero %s.\n", filename);
        break;
    case -3:
        fprintf(stderr, "Error escibiendo en el fichero %s.\n", filename);
        break;
    }
    fflush(stderr);
}

/*
Lectura del fichero para determinar el numero de filas y muestras (samples)
*/
int readInput(char *filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples;

    contlines = 0;

    if ((fp = fopen(filename, "r")) != NULL)
    {
        while (fgets(line, MAXLINE, fp) != NULL)
        {
            if (strchr(line, '\n') == NULL)
            {
                return -1;
            }
            contlines++;
            ptr = strtok(line, delim);
            contsamples = 0;
            while (ptr != NULL)
            {
                contsamples++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;
        return 0;
    }
    else
    {
        return -2;
    }
}

/*
Carga los datos del fichero en la estructra data
*/
int readInput2(char *filename, float *data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;

    if ((fp = fopen(filename, "rt")) != NULL)
    {
        while (fgets(line, MAXLINE, fp) != NULL)
        {
            ptr = strtok(line, delim);
            while (ptr != NULL)
            {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    }
    else
    {
        return -2; // No file found
    }
}

/*
Escribe en el fichero de salida la clase a la que perteneces cada muestra (sample)
*/
int writeResult(int *classMap, int lines, const char *filename)
{
    FILE *fp;

    if ((fp = fopen(filename, "wt")) != NULL)
    {
        for (int i = 0; i < lines; i++)
        {
            fprintf(fp, "%d\n", classMap[i]);
        }
        fclose(fp);

        return 0;
    }
    else
    {
        return -3; // No file found
    }
}

/*
Copia el valor de los centroides de data a centroids usando centroidPos como
mapa de la posicion que ocupa cada centroide en data
*/
void initCentroids(const float *data, float *centroids, int *centroidPos, int samples, int K)
{
    int i;
    int idx;
    for (i = 0; i < K; i++)
    {
        idx = centroidPos[i];
        memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
    }
}

/*
Calculo de la distancia euclidea
*/
float euclideanDistance(float *point, float *center, int samples)
{
    float dist = 0.0;
    for (int i = 0; i < samples; i++)
    {
        dist += (point[i] - center[i]) * (point[i] - center[i]);
    }
    dist = sqrt(dist);
    return (dist);
}

int classifyPoints(float *data, float *centroids, int *classMap,
                   int lines, int samples, int K, int rank)
{

    int i, j;
    int class;
    float dist, minDist;
    int changes = 0;
    for (i = 0; i < lines; i++)
    {
        class = 1;
        minDist = FLT_MAX;
        for (j = 0; j < K; j++)
        {
            dist = euclideanDistance(&data[i * samples],
                                     &centroids[j * samples],
                                     samples);

            if (dist < minDist)
            {
                minDist = dist;
                class = j + 1;
            }
        }
        if (classMap[i] != class)
        {
            changes++;
        }
        classMap[i] = class;
    }

    // Combinar cambios de todos los procesos
    int total_changes;
    MPI_Allreduce(&changes, &total_changes,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    changes = total_changes;

    return (changes);
}

float recalculateCentroids(float *data, float *centroids, int *classMap,
                           int lines, int samples, int K)
{
    int class, i, j;
    int *pointsPerClass;
    pointsPerClass = (int *)calloc(K, sizeof(int));
    float *auxCentroids;
    auxCentroids = (float *)calloc(K * samples, sizeof(float));
    float *distCentroids;
    distCentroids = (float *)malloc(K * sizeof(float));
    if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
    {
        fprintf(stderr, "Error alojando memoria\n");
        exit(-4);
    }

    // Calcular puntos por clase y centroides parciales para el subconjunto de datos asignado a este proceso
    for (i = 0; i < lines; i++)
    {
        class = classMap[i];
        pointsPerClass[class - 1] = pointsPerClass[class - 1] + 1;
        for (j = 0; j < samples; j++)
        {
            auxCentroids[(class - 1) * samples + j] += data[i * samples + j];
        }
    }

    // Combinar puntos por clase y centroides parciales de todos los procesos
    int *global_pointsPerClass = (int *)calloc(K, sizeof(int));
    float *global_auxCentroids = (float *)calloc(K * samples, sizeof(float));
    MPI_Allreduce(pointsPerClass, global_pointsPerClass,
                  K, MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(auxCentroids, global_auxCentroids,
                  K * samples,
                  MPI_FLOAT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    memcpy(pointsPerClass, global_pointsPerClass,
           K * sizeof(int));
    memcpy(auxCentroids, global_auxCentroids,
           K * samples * sizeof(float));
    free(global_pointsPerClass);
    free(global_auxCentroids);

    // Calcular nuevos centroides globales
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < samples; j++)
        {
            auxCentroids[i * samples + j] /= pointsPerClass[i];
        }
    }

    float maxDist = FLT_MIN;
    for (i = 0; i < K; i++)
    {
        distCentroids[i] = euclideanDistance(&centroids[i * samples],
                                             &auxCentroids[i * samples],
                                             samples);
        if (distCentroids[i] > maxDist)
        {
            maxDist = distCentroids[i];
        }
    }

    // Combinar maxDist de todos los procesos
    float global_maxDist=0;
    MPI_Allreduce(&maxDist, &global_maxDist,
                  1,
                  MPI_FLOAT,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    maxDist = global_maxDist;

    memcpy(centroids, auxCentroids,
           (K * samples * sizeof(float)));
    free(distCentroids);
    free(pointsPerClass);
    free(auxCentroids);
    return (maxDist);
}


int main(int argc, char *argv[])
{
    // Inicializar entorno de MPI
    MPI_Init(&argc, &argv);

    // Obtener el número de procesos y el rango del proceso actual
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // START CLOCK***************************************
    clock_t start, end;
    start = MPI_Wtime();

    if (argc != 7)
    {
        fprintf(stderr, "EXECUTION ERROR KMEANS Iterative: Parameters are not correct.\n");
        fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
        fflush(stderr);
        exit(-1);
    }

    // Lectura de los datos de entrada
    //  lines = numero de puntos;  samples = numero de dimensiones por punto
    int lines = 0, samples = 0;

    // Leer datos de entrada en el proceso 0
    if (rank == 0)
    {
        int error = readInput(argv[1], &lines, &samples);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            exit(error);
        }
    }

// Distribuir datos a todos los procesos
    MPI_Bcast(&lines, 1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&samples, 1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    // Asignar memoria para data después de conocer los valores de lines y samples
    float *data = (float*)calloc(lines*samples,sizeof(float));
    if(rank==0){
        int error=readInput2(argv[1],data);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            exit(error);
        }
    }
    int *sendcounts = (int*)malloc(num_procs * sizeof(int));
    int *displs = (int*)malloc(num_procs * sizeof(int));
    int rem = lines % num_procs;
    int sum = 0;
    for (int i = 0; i < num_procs; i++) {
        sendcounts[i] = (lines / num_procs) * samples;
        if (rem > 0) {
            sendcounts[i] += samples;
            rem--;
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }
    float *sub_data = (float*)malloc(sendcounts[rank] * sizeof(float));
    MPI_Scatterv(data, sendcounts, displs, MPI_FLOAT,
                sub_data, sendcounts[rank], MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // prametros del algoritmo. La entrada no esta valdidada
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    // poscion de los centroides en data
    int *centroidPos = (int *)calloc(K, sizeof(int));
    float *centroids = (float *)calloc(K * samples, sizeof(float));
    int *classMap = (int *)calloc(sendcounts[rank] / samples, sizeof(int));
    // Otras variables
    float distCent;
    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
    {
        fprintf(stderr, "Error alojando memoria\n");
        exit(-4);
    }
    int it = 0;
    int changes = 0;

    // Centroides iniciales
    srand(0);
    int i;
    for (i = 0; i < K; i++)
        centroidPos[i] = rand() % lines;

    // Carga del array centroids con los datos del array data
    // los centroides son puntos almacenados en data
    if (rank == 0){
        initCentroids(data, centroids, centroidPos, samples, K);
    }

    // Distribuir centroides iniciales a todos los procesos
    MPI_Bcast(centroids, K * samples,
              MPI_FLOAT,
              0,
              MPI_COMM_WORLD);

    // Resumen de datos caragos
    if (rank == 0){
        printf("\n\tFichero de datos: %s \n\tPuntos: %d\n\tDimensiones: %d\n", argv[1], lines, samples);
        printf("\tNumero de clusters: %d\n", K);
        printf("\tNumero maximo de iteraciones: %d\n", maxIterations);
        printf("\tNumero minimo de cambios: %d [%g%% de %d puntos]\n", minChanges, atof(argv[4]), lines);
        printf("\tPrecision maxima de los centroides: %f\n", maxThreshold);
    }
   

    // END CLOCK*****************************************
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (rank == 0)
        printf("\nAlojado de memoria: %f segundos\n", (double)end - start);
    fflush(stdout);
    //**************************************************
    // START CLOCK***************************************
    start = MPI_Wtime();
    //**************************************************

    // Dividir el conjunto de datos entre los procesos
    
    do
    {
        it++;
        // Clasificar puntos para el subconjunto de datos asignado a este proceso
        changes = classifyPoints(sub_data, centroids,
                         classMap,
                         sendcounts[rank] / samples, samples, K, rank);
        // Recalcular centroides para el subconjunto de datos asignado a este proceso
        distCent = recalculateCentroids(sub_data,
                                centroids,
                                classMap,
                                sendcounts[rank] / samples,
                                samples, K);

        
        // Combinar centroides parciales de todos los procesos
        float *global_centroids = (float *)calloc(K * samples,
                                                  sizeof(float));
        MPI_Allreduce(centroids, global_centroids,
                      K * samples,
                      MPI_FLOAT,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        for (int i = 0; i < K * samples; i++)
            centroids[i] = global_centroids[i] / num_procs;
        free(global_centroids);

        if (rank == 0)
            printf("\n[%d] Cambios de cluster: %d\tMax. dist. centroides: %f",
                   it,
                   changes,
                   distCent);
    } while ((changes > minChanges) && (it < maxIterations) && (distCent > maxThreshold));

    // Condiciones de fin de la ejecucion
    if (rank == 0){
        if (changes <= minChanges )
            {
                printf("\n\nCondicion de parada: Numero minimo de cambios alcanzado: %d [%d]", changes, minChanges);
            }
        else if (it >= maxIterations)
            {
                printf("\n\nCondicion de parada: Numero maximo de iteraciones alcanzado: %d [%d]", it, maxIterations);
            }
        else
            {
                printf("\n\nCondicion de parada: Precision en la actualizacion de centroides alcanzada: %g [%g]", distCent, maxThreshold);
            }
    }
    
    // Escritura en fichero de la clasificacion de cada punto
    //  Escribir resultado en el proceso 0
    
    int *recvcounts = (int*)malloc(num_procs * sizeof(int));
    rem = lines % num_procs;
    sum = 0;
    for (int i = 0; i < num_procs; i++) {
        recvcounts[i] = lines / num_procs;
        if (rem > 0) {
            recvcounts[i]++;
            rem--;
        }
        displs[i] = sum;
        sum += recvcounts[i];
    }
    


    int *global_classMap = (int*)malloc(lines * sizeof(int));
    MPI_Gatherv(classMap, recvcounts[rank], MPI_INT,
            global_classMap, recvcounts, displs, MPI_INT,
            0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        int error = writeResult(global_classMap, lines, argv[6]);
        if (error != 0)
        {
            showFileError(error, argv[6]);
            exit(error);
        }
    }

    
    // END CLOCK*****************************************

    // Sincronizar todos los procesos antes de medir el tiempo y liberar la memoria
    MPI_Barrier(MPI_COMM_WORLD);

    end = MPI_Wtime();
    if (rank == 0)
        printf("\nComputacion: %f segundos", (double)end - start);
    fflush(stdout);

    //**************************************************
    // START CLOCK***************************************
    start = MPI_Wtime();
    //**************************************************

    // Liberar memoria dinámica
    free(data);
    free(classMap);
    free(centroidPos);
    free(centroids);
    free(global_classMap);
    free(recvcounts);
    free(sendcounts);
    free(displs);

    // END CLOCK*****************************************

    // Sincronizar todos los procesos antes de medir el tiempo y liberar la memoria
    MPI_Barrier(MPI_COMM_WORLD);

    end = MPI_Wtime();
    if (rank==0)
        printf("\n\nLiberacion: %f segundos\n", (double)end - start);
    fflush(stdout);

    // Finalizar entorno de MPI
    MPI_Finalize();

    return 0;
}
