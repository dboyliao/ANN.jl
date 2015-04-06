import Base.show
import Base.repr

type NeuralLayer
    weight::Union(Matrix{FloatingPoint}, Array)
    _node_values::Array{FloatingPoint, 1}
    
    function NeuralLayer(w, nv)
        new(w, nv)
    end
end

type SimpleNeuralNetwork
    structure::Vector
    act_function::Function
    act_diffunction::Function
    layers::Array{NeuralLayer, 1}
    
    # Constructor for NeuralNetwork type
    function SimpleNeuralNetwork(struct, act_fun, act_diff = None)
        if act_diff == None
            act_diff = derivative(act_fun)
        end
        layers = NeuralLayer[]
        for ind = 1:(length(struct)-1)
            dim_in = struct[ind]
            dim_out = struct[ind+1]
            b = sqrt(6) / sqrt(dim_in + dim_out)
            w = 2b*rand(dim_out, dim_in + 1) - b
            node_values = push!([1.0], rand(dim_in)...)
            temp_layer = NeuralLayer(w, node_values)
            push!(layers, temp_layer)
        end
        # append the output layer.
        w = []
        node_values = rand(struct[end])
        temp_layer = NeuralLayer(w, node_values)
        push!(layers, temp_layer)
        nn = new(struct, act_fun, act_diff, layers)
    end
    
end

## Numeric derivative.
function derivative(func::Function, epsilon = 1e-8)
    function (x)
        (func(x + epsilon) - func(x - epsilon))/(2*epsilon)
    end
end

function show(nn::SimpleNeuralNetwork)
    struct = join([string(i) for i in nn.structure], "x")
    println("It is a ", struct, " simple neural network.")
    println("Activate Function: ", nn.act_function)
    println()
end

function repr(nn::SimpleNeuralNetwork)
    struct = join([string(i) for i in nn.structure], "x")
    msg = join(["It is a ", struct, ".\n"] , "")
    msg = join([msg, "Activate Function: ", string(nn.act_function)], "")
    msg
end

function predict(nn::SimpleNeuralNetwork, data::Matrix)
    predict_results = Array{Float64, 1}[]
    for data_id = 1:size(data)[1]
        v = [x for x in data[data_id, :]]
        forward_prob!(nn, v)
        push!(predict_results, nn.layers[end]._node_values)
    end
    predict_results
end

function forward_prob!(nn::SimpleNeuralNetwork, x::Vector)
    nn.layers[1]._node_values = [1, x]
    for layer_id = 1:(length(nn.structure)-1)
        temp = nn.layers[layer_id].weight * nn.layers[layer_id]._node_values
        nn.layers[layer_id + 1]._node_values = [1., temp]
    end
    nn.layers[end]._node_values = nn.layers[end]._node_values[2:end]
    return
end


function fit!(nn::SimpleNeuralNetwork, data::Matrix)
    println("Not yet implemented.")
end