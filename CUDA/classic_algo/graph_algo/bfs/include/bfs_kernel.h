#ifndef BFS_KERNEL_H
#define BFS_KERNEL_H

//define graph
struct Graph {
    int numVertices;
    int numEdges;
    int* offsets;      
    int* adjacency; 
};




#endif
