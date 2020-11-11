from .network_arch import RelationNet, PlainCNN

def build_network(config):
    if config.network.type == 'RelationNet':
        net = RelationNet(output_dim=config.task.action_dim)
    elif config.network.type == 'PlainCNN':
        net = PlainCNN(output_dim=config.task.action_dim)
    else:
        raise NotImplementedError()
    net.cuda()
    return net
