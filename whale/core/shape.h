#ifndef CUBASE_SHAPE_H
#define CUBASE_SHAPE_H

#include <unistd.h>
#include <assert.h>
#include <vector>
#include <chrono>
#include <random>

#include <cuda_runtime.h>
#include <cuda.h>
#include "timer.h"

template<int Dim>
struct Shape {
    Shape() {}

    explicit Shape(std::vector<int>& dims) {
        init(dims, dims.size());
    }

    Shape(const Shape<Dim>& other) {
        (*this) = other;
    }

    Shape<Dim>& operator=(const Shape<Dim>& other) {
        _front = other._front;
        _ends = other._ends;
        return *this;
    }
    
    inline int dim() {
        return Dim;
    }

    inline int count() {
        return _front*_ends.count();
    }
    inline Shape<Dim-1>& ends() {
        return _ends;
    }

    int& operator[](int index) {
        assert(0 <= index && index < Dim);
        if(index == 0) {
            return _front;
        }
        return _ends[index-1];
    }

    bool operator==(Shape<Dim>& other) {
        if(Dim != other.dim() || _front != other[0]) {
            return false;
        }
        return (_ends == other._ends);
    }

    bool operator!=(Shape<Dim>& other) {
        return !((*this) == other);
    }

    void init(const std::vector<int>& dims, size_t size) {
        assert(size <= dims.size() && size > 0);
        _front = dims[dims.size() - size];
        _ends.init(dims, size-1);
    }    

private:
    int _front;
    Shape<Dim-1> _ends;
};

template<>
struct Shape<1> {
    Shape() {}
    explicit Shape(int dim) {
        _front = dim;
    }

    inline int dim() {
        return 1;
    }

    inline int count() {
        return _front;
    } 
    
    int& operator[] (int index) {
        if(index != 0) {
            printf("ERROR: index overflow");
            exit(1);
        }
        return _front;
    }

    bool operator==(Shape<1>& other) {
        if(_front != other._front) {
            return false;
        }
        return true;
    }

    bool operator!=(Shape<1>& other) {
        return !((*this) == other);
    }

    void init(const std::vector<int>& dims, size_t size) {
        assert(size == 1);
        _front = dims[dims.size() - size];
    }

private:
    int _front;
};

/*
template<typename T>
void transform(T* data, Shape<2>& shape, std::vector<int> order) { 
    T* swap_data = new T[shape.count()];
    auto get_new_idx = [&](int i, int j, int dim) -> int {
        if(order[dim] == 0) {
            return i;
        } else if(order[dim] == 1) {
            return j;
        } else {
          fprintf(stderr, "input dim(%d) overflow!", dim);
          exit(1);
        }
    };
    Shape<2> new_shape;
    new_shape[0] = get_new_idx(shape[0], shape[1], 0);
    new_shape[1] = get_new_idx(shape[0], shape[1], 1);
 
    for(int i=0; i < shape[0]; i++) {
        for(int j=0; j < shape[1]; j++) {
            T& src_value = data[i*shape.ends().count() + j];
            swap_data[get_new_idx(i, j, k, l, 0) * new_shape.ends().count() + 
                      get_new_idx(i, j, k, l, 1)] = src_value;
        }
    }
} 


template<typename T>
void transform(T* data, Shape<3>& shape, std::vector<int> order) { 
    T* swap_data = new T[shape.count()];
    auto get_new_idx = [&](int i, int j, int k, int dim) -> int {
        if(order[dim] == 0) {
            return i;
        } else if(order[dim] == 1) {
            return j;
        } else if(order[dim] == 2) {
            return k;
        } else {
          fprintf(stderr, "input dim(%d) overflow!", dim);
          exit(1);
        }
    }
    Shape<3> new_shape;
    new_shape[0] = get_new_idx(shape[0], shape[1], shape[2], 0);
    new_shape[1] = get_new_idx(shape[0], shape[1], shape[2], 1);
    new_shape[2] = get_new_idx(shape[0], shape[1], shape[2], 2);
 
    for(int i=0; i < shape[0]; i++) {
        for(int j=0; j < shape[1]; j++) {
	    for(int k=0; k < shape[2]; k++) {
                T& src_value = data[i*shape.ends().count() + 
                                    j*shape.ends().ends().count() + k];
                swap_data[get_new_idx(i, j, k, l, 0) * new_shape.ends().count() + 
                          get_new_idx(i, j, k, l, 1) * new_shape.ends().ends().count() +
                          get_new_idx(i, j, k, l, 2)] = src_value;
            }
        }
    }
} 

template<typename T>
void transform(T* data, Shape<4>& shape, std::vector<int> order) { 
    T* swap_data = new T[shape.count()];
    auto get_new_idx = [&](int i, int j, int k, int l, int dim) -> int {
        if(order[dim] == 0) {
            return i;
        } else if(order[dim] == 1) {
            return j;
        } else if(order[dim] == 2) {
            return k;
        } else if(order[dim] == 3) {
            return l
        } else {
          fprintf(stderr, "input dim(%d) overflow!", dim);
          exit(1);
        }
    }
    Shape<4> new_shape;
    new_shape[0] = get_new_idx(shape[0], shape[1], shape[2], shape[3], 0);
    new_shape[1] = get_new_idx(shape[0], shape[1], shape[2], shape[3], 1);
    new_shape[2] = get_new_idx(shape[0], shape[1], shape[2], shape[3], 2);
    new_shape[3] = get_new_idx(shape[0], shape[1], shape[2], shape[3], 3);
 
    for(int i=0; i < shape[0]; i++) {
        for(int j=0; j < shape[1]; j++) {
	    for(int k=0; k < shape[2]; k++) {
                for(int l=0; l < shape[3]; l++) {
                    T& src_value = data[i*shape.ends().count() + 
                                        j*shape.ends().ends().count() + 
                                        k*shape.ends().ends().ends().count() + l];
                    swap_data[get_new_idx(i, j, k, l, 0) * new_shape.ends().count() + 
                              get_new_idx(i, j, k, l, 1) * new_shape.ends().ends().count() +
                              get_new_idx(i, j, k, l, 2) * new_shape.ends().ends().ends().count() +
                              get_new_idx(i, j, k, l, 3)] = src_value;
                }
            }
        }
    }
} */

#endif
