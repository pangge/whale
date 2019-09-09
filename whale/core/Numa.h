/**
 * @brief reference to http://man7.org/linux/man-pages/man7/numa.7.html 
 *				   and http://man7.org/linux/man-pages/man3/numa.3.html
 */

#ifndef NUMA_H
#define NUMA_H

#include <numa.h>
#include <unistd.h>

size_t chNUMAgetPageSize()
{
  return (size_t) sysconf(_SC_PAGESIZE);
}

bool chNUMAnumNodes( int *p )
{
    if ( numa_available() != -1 ) {
        *p = numa_max_node() + 1;
        return true;
    }
    return false;
}

void * chNUMApageAlignedAlloc( size_t bytes, int node )
{
    void *ret;
    ret = numa_alloc_onnode( bytes, node );
    return ret;
}

void chNUMApageAlignedFree( void *p, size_t bytes )
{
    numa_free( p, bytes );
}


#include <stdint.h>

// Portable implementations that use the functions we just defined
bool chNUMApageAlignedAllocHost( void **pp, size_t bytes, int node )
{
    bytes += chNUMAgetPageSize();
    void *p = chNUMApageAlignedAlloc( bytes, node );
    if ( NULL == p )
        goto Error;
    if ( cudaSuccess !=  cudaHostRegister( p, bytes, 0 ) )
        goto Error;
    *((size_t *) p) = bytes;
    *pp = (void *) ((char *) p+chNUMAgetPageSize());
    return true;
Error:
    if ( p ) {
        cudaHostUnregister( p );
        chNUMApageAlignedFree( p, bytes );
    }
    return false;
}

void chNUMApageAlignedFreeHost( void *p )
{
    p = (void *) ((uintptr_t) p-chNUMAgetPageSize() );
    size_t bytes = *(size_t *) p;
    cudaHostUnregister( p );
    chNUMApageAlignedFree( p, bytes );
}


#endif // NUMA_H
