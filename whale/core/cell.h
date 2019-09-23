#ifndef WHALE_CELL_H
#define WHALE_CELL_H

#include "core/timer.h"
#include "core/shape.h"
#include "core/types.h"
#include "core/buffer.h"

namespace whale {

template<typename T, int Dim>
class Cell {
public:
    Cell(Layout layout = Layout::NCHW):_layout(layout) {}

    explicit Cell(const std::vector<int>& dims, Layout layout = Layout::NCHW) {
	    _layout = layout;
        _shape.init(dims, dims.size());
        Alloc(_shape);
    }

    explicit Cell(const Shape<Dim>& shape, Layout layout = Layout::NCHW) {
	_layout = layout;
        Alloc(shape);
    }

    ~Cell() { _free(); }

public:

    template<typename functor, typename ...ArgTs>
    void map_cpu(functor funcs, ArgTs ...args) {
        for(int i=0; i < size(); i++) {
            funcs(h_data()[i], args...);
        }
    }
    
    template<typename functor, typename ...ArgTs>
    void map_all(functor funcs, ArgTs ...args) {
        map_cpu(funcs, args...);
        sync<GPU>();
    }

    /*template<Layout::type LayoutT>
    void transform(int cx = 4){
        transform<LayoutT>(_h_data, _shape, std::vector<int> order);
    }*/

    template<DeviceT DevT>
    void sync() {
        switch(DevT) {
            case GPU: {
                cuda(Memcpy(_d_data, h_data(), size()*sizeof(T), cudaMemcpyHostToDevice)); 
            } break;
            case CPU: {
                cuda(Memcpy(h_data(), _d_data, size()*sizeof(T), cudaMemcpyDeviceToHost));
            } break;
            default: {
                printf("ERROR: DevT(%d) not known!", DevT);                
                exit(1);
            }
        }
    }

    template<DeviceT DevT>
    void CopyFrom(Cell<T, Dim>& operand) {
        switch(DevT) {
            case GPU: {
                assert(size() == operand.size());
                cuda(Memcpy(_d_data, operand.d_data(), size()*sizeof(T), cudaMemcpyDeviceToDevice));
            } break;
            case CPU: {
                printf("ERROR: cpu to cpu copy not support yet!");
                exit(1);
            } break;
            default: { 
                printf("ERROR: DevT(%d) not known!", DevT); 
                exit(1); 
            }
        }
    }

    void Alloc(const Shape<Dim>& shape) {
        _shape = shape;
        _h_data.resize(_shape.count());
        _free();
        cuda(GetDevice(&_device_id));
        cuda(Malloc(&_d_data, _shape.count()*sizeof(T)));
    }

public:
    inline Shape<Dim>& shape() { return _shape; }

    inline int dim() { return Dim; }

    inline size_t size() { return _shape.count(); }

    T*  data() { return _data_ptr.get()->get(); }

    const T* data() const { return _data_ptr.get()->get(); }

//private:
//    void _free() {
//        if(_d_data) { 
//            //std::cout<<"_free (" << size() * sizeof(T) / 1e6 << " MB) mem on gpu.\n";
//            cuda(Free(_d_data)); 
//        }
//    }

private:
    Shape<Dim> _shape;
    Layout _layout;
	std::shared_ptr<Buffer> _data_ptr;
    //std::vector<T> _h_data;
    //T* _d_data{nullptr};
};

///////////////////////////////////  single value op functor ///////////////////////////

/* only for local data type */
template<typename T>
void random_gen(T& local_value, T min, T max) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<T> dis(min, max);
    T random_num = dis(gen);
    local_value = random_num;
}

template<typename T>
void fill_value(T& local_value, T value) {
    local_value = value;
}

}

#endif
