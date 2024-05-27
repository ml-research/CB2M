from template_model import MLP, inception_v3, End2EndModel, BottleneckMLP


# Concept predictor for Independent Model (CUB)
def bottleneck_model(pretrained, num_classes, use_aux, n_attributes):
    return inception_v3(pretrained=pretrained, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=True)


# bottleneck for MNIST
def mnist_bottleneck(n_attributes):
    return BottleneckMLP(n_attributes)


# bottleneck for CMNIST
def cmnist_bottleneck(n_attributes):
    return BottleneckMLP(n_attributes, input_size=2352)


# Independent Model
def independent_model(n_attributes, num_classes):
    # X -> C part is separate, this is only the C -> Y part
    return MLP(input_dim=n_attributes, num_classes=num_classes)


# Joint Model
def joint_model(pretrained, num_classes, use_aux, n_attributes, use_sigmoid):
    model1 = inception_v3(pretrained=pretrained, num_classes=num_classes, aux_logits=use_aux,
                          n_attributes=n_attributes, bottleneck=True)
    model2 = MLP(input_dim=n_attributes, num_classes=num_classes)
    return End2EndModel(model1, model2, use_sigmoid)

