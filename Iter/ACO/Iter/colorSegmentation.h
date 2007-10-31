#define MAX_CLUSTERS 30
#define MAX_TRAINING_PIXELS 20000


void initPatterns(int,int);
void kMeansClustering (IplImage* src, int k);
void mostrarPatrones(int,IplImage*);
void patronesAprendizaje(int,int);
void mahalanobisDistance(IplImage*,int);