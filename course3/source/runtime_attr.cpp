#include "runtime/runtime_attr.hpp"


namespace kuiper_infer{
    void RuntimeAttribute::clearWeight(){
        if(!this->weight_data.empty()){
            std::vector<char> tmp = std::vector<char>();
            this->weight_data.swap(tmp);
        }
    }
}