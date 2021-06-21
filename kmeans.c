/**
   kmeans.c implements the K-means Algorithm as shown in class.
**/

#define PY_SSIZE_T_CLEAN
#define MAX_ITER 300
#include <Python.h>

int N, k, d;

static double ** getInput(PyObject * vectors);
static double dist(double * vec1, double * vec2);
static int * kmeans(PyObject * observations, PyObject * firsts);
static int belongs (double ** centroids, double * vec);
static double * addVectors(double * vec1, double * vec2);
static double * centroidCalc(double * vec, int member);
static PyObject * kmeanspp(PyObject *self, PyObject *args);

/** Integration with Python.
Input from python: Observations, k centroids to begin with, k, n and d
Output to Python: an array of clustering of each observation from input, after running the clustering algorithm on them
**/

static PyObject * kmeanspp(PyObject *self, PyObject *args)
{
    PyObject *vectors, *centroids, *vector1, *vector2;
    int *clusters;
    Py_ssize_t i;
    if(!PyArg_ParseTuple(args, "OOiii:Problem with accepting arguments from Python", &vectors, &centroids, &k, &N, &d)) {
        PyErr_SetString(PyExc_Exception, "There was a problem with the input!");
        return NULL;
    }
    /* Is it a list? */
    if (!PyList_Check(vectors) || !PyList_Check(centroids)){
        PyErr_SetString(PyExc_Exception, "There was a problem with the input!");
        return NULL;
    }
    for (i = 0; i < PyList_Size(vectors); i++) {
        vector1 = PyList_GetItem(vectors, i);
        if (i < PyList_Size(centroids)) {
            vector2 = PyList_GetItem(vectors, i);
            if (!PyList_Check(vector2)) {
                PyErr_SetString(PyExc_Exception, "There was a problem with the input - it's not a list of lists!");
                return NULL;
            }
        }
        if (!PyList_Check(vector1)) {  /* We only print lists */
                PyErr_SetString(PyExc_Exception, "There was a problem with the input - it's not a list of lists!");
                return NULL;
            }
    }
    clusters = kmeans(vectors, centroids);
    if (clusters == NULL) {
    return NULL;
    }
    PyObject * lst = PyList_New(N);
    if (!lst)
        return NULL;
    for (i = 0; i < N; i++) {
        PyList_SET_ITEM(lst, i, Py_BuildValue("i", clusters[i]));
    }
    free(clusters);
    return lst;
}

/** This function calculates the square distance between two vectors.
params: vec1, vec2 - two vectors, represented by one-dimensional C arrays
returns: counter - the square distance between the vectors given
**/

static double dist(double * vec1, double * vec2){
    double counter;
    int i;
    counter = 0;
    for(i = 0; i < d; i++){
        counter += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return counter;
}

/** This function calculates the closest centroid to a vector given and returns its cluster index.
params: centroids - a two-dimensional C array of centroids /
        vec - a vector represented by a one-dimensional C array
returns: minInd - the relevant clusters' index, determined by finding the closest centroid to the vector given
**/

static int belongs(double ** centroids, double * vec){
    int minInd, i;
    double minDist, currDist;
    minInd = 0;
    minDist = dist(centroids[0], vec);
    for (i = 1; i < k; i++){
        currDist = dist(centroids[i], vec);
        if (currDist < minDist){
            minInd = i;
            minDist = currDist;
        }

    }
    return minInd;
}

/** This function sums two vectors.
params: vec1, vec2 - two vectors represented by one-dimensional C arrays
returns: vec1 - the result of the sum of the two vectors, represented by a one-dimensional C array
**/

static double * addVectors(double * vec1, double * vec2){
    int i;
    for(i = 0; i < d; i++){
        vec1[i] += vec2[i];
    }
    return vec1;

}

/** This function calculates the updated centroids for the data given.
params: vec - the result of sums of the different vectors in this cluster, represented by a one-dimensional C array \
        member - the number of different vectors in this cluster
returns: vec - the updated centroid, calculated by the division of the sum given by the number of vectors
**/

static double * centroidCalc(double * vec, int member){
    int i;
    for(i = 0; i < d; i++){
        vec[i] = (vec[i] / member);
    }
    return vec;
}

/** This function converts PyObject arrays to C arrays.
params: vectors - a PyObject array of vectors
returns: newVecs - a two-dimensional C array of the same vectors
**/

static double ** getInput(PyObject * vectors) {
    double ** newVecs = (double **) calloc (PyList_Size(vectors), sizeof(double *));
    if (newVecs == NULL) {
        printf("Couldn't allocate space!");
        return NULL;
    }
    PyObject * item;
    PyObject * inside;
    Py_ssize_t i, j, l;
    for (i = 0; i < PyList_Size(vectors); i++) {
        item = PyList_GetItem(vectors, i);
        newVecs[i] = (double *) calloc(PyList_Size(item), sizeof(double));
        if (newVecs[i] == NULL) {
                printf("Couldn't allocate space!");
                for (l = 0; l < i; l++) {
                    free(newVecs[l]);
                }
                free(newVecs);
                return NULL;
            }
        for (j = 0; j < PyList_Size(item); j++) {
            inside = PyList_GetItem(item, j);
            if (!PyFloat_Check(inside)) {
                for (l = 0; l < i; l++) {
                    free(newVecs[l]);
                }
                free(newVecs);
                printf("Problem with values in the input!");
                return NULL;
            }
            newVecs[i][j] = PyFloat_AsDouble(inside);
        }
    }
    return newVecs;
}

/** This function clusters the observations given using the K-Means algorithms shown in class.
Params: observations - a PyObject array of the observations \
        firsts - a PyObject array of the initial clusters
Returns: clusters - a one-dimensional array of clusters, with each observation clustered to a specific cluster **/

/**
Disclaimer: with asserts, this function is exactly 60 lines long. Since we didn't want to take a risk with memory leaks - it is now two times longer.
Plus - we tried using epsilon instead of an absolute zero, \
but the results became drastically worse so we decided to stick with zeros instead to perform better clustering.
**/

static int * kmeans(PyObject * observations, PyObject * firsts) {
    int ind, changed, g, n, m, y, i, j, f, p, a;
    double ** vectors = getInput(observations);
    if (vectors == NULL) {
        return NULL;
    }
    double ** initials = getInput(firsts);
    if (initials == NULL) {
        for (i = 0; i < N; i++) {
            free(vectors[i]);
        }
        free(vectors);
    }
    int * members;
    int * clusters;
    double ** sumVectors;
    ind = 0;
    changed = 0;
    clusters = (int *) calloc(N, sizeof(int));
    if (clusters == NULL) {
        printf("Couldn't allocate space!");
        for (i = 0; i < N; i++) {
            free(vectors[i]);
        }
        free(vectors);
        for (i = 0; i < k; i++) {
            free(initials[i]);
        }
        free(initials);
        return NULL;
    }
    members = (int *) calloc(k, sizeof(int));
    if (members == NULL) {
        printf("Couldn't allocate space!");
        free(clusters);
        for (i = 0; i < N; i++) {
            free(vectors[i]);
        }
        free(vectors);
        for (i = 0; i < k; i++) {
            free(initials[i]);
        }
        free(initials);
        return NULL;
    }
    sumVectors = (double **) malloc(k * sizeof(double *));
    if (sumVectors == NULL) {
        printf("Couldn't allocate space!");
        free(clusters);
        free(members);
        for (i = 0; i < N; i++) {
            free(vectors[i]);
        }
        free(vectors);
        for (i = 0; i < k; i++) {
            free(initials[i]);
        }
        free(initials);
        return NULL;
    }
    for (g = 0; g < k; g++) {
        sumVectors[g] = (double *) calloc(d, sizeof(double));
        if (sumVectors[g] == NULL) {
            printf("Couldn't allocate space!");
            free(clusters);
            free(members);
            for (a = 0; a < g; a++) {
                free(sumVectors[a]);
            }
            free(sumVectors);
            for (i = 0; i < N; i++) {
            free(vectors[i]);
            }
            free(vectors);
            for (i = 0; i < k; i++) {
                free(initials[i]);
            }
            free(initials);
            return NULL;
            }
        }
    while(ind < MAX_ITER && changed == 0) {
        for (n = 0; n < N; n++) {
            int belong = belongs(initials, vectors[n]);
            clusters[n] = belong;
            members[belong] += 1;
            sumVectors[belong] = addVectors(sumVectors[belong], vectors[n]);
        }
        changed = 1;
        for (m = 0; m < k; m++) {
            double * currCent = centroidCalc(sumVectors[m], members[m]);
            for (f = 0; f < d; f++) {
                if (currCent[f] != initials[m][f]) {
                    changed = 0;
                    break;
                }
            }
            for (y = 0; y < d; y++) {
                initials[m][y] = currCent[y];
            }
        }
        for (j = 0; j < k; j++) {
            members[j] = 0;
            for(i = 0; i < d; i++){
                sumVectors[j][i] = 0;
            }
        }
        ind += 1 ;
    }
    free(members);
    for (p = 0; p < N; p++) {
        free(vectors[p]);
        if (p < k) {
            free(sumVectors[p]);
            free(initials[p]);
        }
    }
    free(sumVectors);
    free(vectors);
    free(initials);
    return clusters;
}


static PyMethodDef myMethods[] = {
        {"kmeanspp", (PyCFunction) kmeanspp, METH_VARARGS, PyDoc_STR("Kmeanspp function")},
        {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        myMethods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&moduledef);
}