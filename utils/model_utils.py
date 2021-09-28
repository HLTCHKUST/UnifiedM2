import pdb
import re

def layer_freeze(model, num_layers_to_freeze):
    '''
    freeze num_layers_to_freeze initial layers of RoBERTa
    '''
    # assert parameters are same as named_parameters
    num_params = 0
    num_named_params = 0
    for param in model.parameters():
        num_params += 1
    for param in model.named_parameters():
        num_named_params += 1
    try:
        assert num_params == num_named_params
    except AssertionError:
        print("Error: Expected all parameters of {} to be named!".format(type(model)))

    layers_frozen = {i: [] for i in range(num_layers_to_freeze)}
    for name, param in model.named_parameters():
        if re.match('encoder.layer.*', name):
            try:
                layer = int(name[len('encoder.layer.'):len('encoder.layer.')+2])
            except ValueError:
                layer = int(name[len('encoder.layer.'):len('encoder.layer.')+1])
            # is an encoder layer
            if layer < num_layers_to_freeze:
                # if name is first few layers
                layers_frozen[layer].append(name)
                param.requires_grad = False
    print("Froze layers {}".format(list(layers_frozen.keys())))

