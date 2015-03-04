#include <stdio.h>
#include <omp.h>
#include <mpi.h>

void ** pbsearch(const void *keys, size_t num_keys, const void *base, size_t nmemb, size_t size,
                 int (*compar)(const void *, const void *),int t)
{
// keys:     pointer to the begining of the list of keys to search
// num_keys: number of keys to search
// base:     searching environment, i.e. the array
// nmemb:    number of elements in the array
// size:     the size of each element in base
// t:        number of threads
// *compar:  the comparing function refered to


    const void *plast, *pfirst;
    pfirst=(void *)( (const char *)base );
    plast= (void *)( (const char *)base + (nmemb-1)*size );

    int complast, compfirst, k;
    const void *key;
    const void **positions = malloc(sizeof(void*)*num_keys);// to store the positions of the keys
    size_t global_l, global_u, local_l, local_u;
    const void *p, *p1, *p2;
    int comp, comp1, comp2;
    int i;
    int delta, np, residue; //parameters for data partitioning   
    int goreturn, gofor;    //flag the status for finding the element: 
                            //goreturn=1: key is found; gofor=1: key is in one of the subarrays
    for (k=0;k<num_keys;k++)
    {
        key=(void *) ((const char *)keys + k*size);
        complast =(*compar)(key,plast);
        compfirst=(*compar)(key,pfirst);
        goreturn=0;gofor=0; // initialize goreturn and gofor
        if (complast>0 || compfirst<0) //if the key is less than the first or greater than the last element
            p=NULL; goreturn=1; // special case

        global_l=0; local_u=nmemb;
        while (global_l<global_u && goreturn==0 )
        {
            gofor=0; np=0; delta=(global_u-global_l)/t; residue=global_u-global_l-delta*t;
            #pragma omp parallel for private(np,i,local_l,local_u,p1,p2,comp1,comp2)
            for (i=0;i<t;i++) //parfor each thread
            {
                /* specify the local lower and upper bounds in each thread  */
                if (gofor || goreturn) {gofor=0;continue;}
                if (i>=residue && delta==0)
                    {continue;} // no data in the thread, do nothing
                else if (i<residue)
                    {local_l=i*(delta+1);local_u=local_l+delta+1;} // add one data into each thread
                else if (i>=residue && delta>0)
                    {local_l=i*delta+residue; local_u=local_l+delta;}

                p1=(void *) ( ((const char *)base) + (global_l+local_l)*size  );// local lower bound
                comp1=(*compar)(key,p1);
                p2=(void *) ( ((const char *)base) + (global_l+local_u-1)*size);//
                comp2=(*compar)(key,p2);
                if (comp1==0) {p=p1; goreturn=1; }
                else if (comp1>0 && comp2<=0){global_u=global_l+local_u; global_l=global_l+local_l; gofor=1;}
                np=omp_get_thread_num();
            }
            if (goreturn) positions[k]=p;
            else positions[k]=NULL;
        }
    }
    return (void **) positions;
}

int compare_ints(const void * a, const void * b ){return *(int*)a-*(int*)b;}

int main (int argc, char *argv[])
{
    int num_mpi, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_mpi);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_1=MPI_Wtime();
    int num_keys=1048576; int n=1000000000;
    int **pItem;
    int k,i;
    int key[num_keys];
    if (rank==0)
        for (k=0;k<num_keys;k++){srand(time(NULL)+k);key[k]=rand()%n;};

    MPI_Bcast(key,num_keys,MPI_INT,0,MPI_COMM_WORLD);
    int num_thread_permpi=1;
    omp_set_num_threads(num_thread_permpi);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_2=MPI_Wtime();

    int npm=(int)n/num_mpi;
    int residue=n-npm*num_mpi;
    int value[npm+1];int found[num_keys];

    for (k=0;k<num_keys;k++)
        found[k]=0;
    if (rank<residue)
    {
        for (i=0;i<npm+1;i++) value[i]=rank*(npm+1)+i;
        pItem=(int **) pbsearch(key,num_keys,value,npm+1,sizeof(int),compare_ints,num_thread_permpi);
    }
    else
    {
        for (i=0;i<npm;i++) value[i]=rank*(npm+1)+i;
        pItem=(int **) pbsearch(key,num_keys,value,npm,sizeof(int),compare_ints,num_thread_permpi);
    }
    for (k=0;k<num_keys;k++)
    {
        if (pItem[k]!=NULL)
        {
            //printf ("%d is in the in the array.\n",key[k]);
            found[k]=1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double end_time_2=MPI_Wtime();
    int global_found[num_keys];
    for (k=0;k<num_keys;k++) global_found[k]=0;
    MPI_Reduce(found,global_found,num_keys,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    /*
    if (rank==0)
    {
        for (k=0;k<num_keys;k++)
            if (global_found[k]==0)
                printf("%d is not in the array.\n",key[k]);
    }
    */
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time_1=MPI_Wtime();
    if (rank==0)
        printf("time elapsed: %f.\n",end_time_2-start_time_2);
    MPI_Finalize();
    return 0;

}
