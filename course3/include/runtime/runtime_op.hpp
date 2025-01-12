#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
// #include "layer/abstract/layer.hpp"
#include "runtime/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace kuiper_infer{
    class Layer;
    struct RuntimeOperator{
        virtual ~RuntimeOperator();

        bool has_forward = false;
        std::string name;  //计算节点的名称
        std::string type;  //计算节点的类型
        std::shared_ptr<Layer> layer; //计算节点对应的层

        std::vector<std::string> output_names; //计算节点的输出名称
        std::shared_ptr<RuntimeOperand> output_operand; //计算节点的输出

        std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands; // 计算节点的输入
        std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq; // 节点输入操作数，顺序排列
        std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators; // 计算节点的输出

        std::map<std::string, RuntimeParameter*> params; //算子的参数信息
        std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute; // 算子的属性信息，内含有权重信息

    
    };
}
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_