#ifndef WHALE_CELL_H
#define WHALE_CELL_H

#include <memory>
#include "core/timer.h"
#include "core/shape.h"
#include "core/types.h"
#include "core/buffer.h"

namespace whale {

template<typename T, int Dim=4>
class Cell {
public:
    Cell(Target target=Target::Default) {
        _layout = LayoutTraits<Dim>::Default;
        _data_ptr = std::make_shared<Buffer<T>>(target, 0);
    }

    explicit Cell(const std::vector<int>& dims, Target target=Target::Default) {
	    _layout = LayoutTraits<Dim>::Default;
        _shape.init(dims, dims.size());
        _data_ptr = std::make_shared<Buffer<T>>(target, size());
    }

    explicit Cell(const Shape<Dim>& shape, Target target=Target::Default) {
	    _layout = LayoutTraits<Dim>::Default;
        _shape = shape;
        _data_ptr = std::make_shared<Buffer<T>>(target, size());
    }

    ~Cell() {}

public:

    template<typename functor, typename ...ArgTs>
    void map_cpu(functor funcs, ArgTs ...args) {
        Target tmp = target();
        _data_ptr->switch_to(Target::Default);
        for(int i=0; i < size(); i++) {
            funcs(data()[i], args...);
        }
        _data_ptr->switch_to(tmp);
    }
    
    /*template<Layout::type LayoutT>
    void transform(int cx = 4){
        transform<LayoutT>(_h_data, _shape, std::vector<int> order);
    }*/

    void to(Target target) { _data_ptr->switch_to(target); }

    int CopyFrom(Cell<T, Dim>& operand) {
        _shape = operand.shape();
        _layout = operand.layout();
        return whale::mem_cpy(*(_data_ptr.get()), *(operand._data_ptr.get()));
    }

    void Alloc(const Shape<Dim>& shape) {
        _shape = shape;
        _data_ptr->realloc(_shape.count());
    }

public:
    inline Shape<Dim>& shape() { return _shape; }
    inline Layout& layout() { return _layout; }

    inline int dim() { return Dim; }

    inline size_t size() { return _shape.count(); }

    inline size_t bytes() { return _data_ptr->bytes(); }

    inline size_t rel_bytes() { return _data_ptr->rel_bytes(); }

    inline Target target() { return _data_ptr->target(); }

    T*  data() { return _data_ptr.get()->get(); }

    const T* data() const { return _data_ptr.get()->get(); }

private:
    Shape<Dim> _shape;
    Layout _layout;
    std::shared_ptr<Buffer<T> > _data_ptr;
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
