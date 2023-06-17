'''
parse args to build model.
'''

from prop import Propagation

def get_model(args, dataset, all_features):

    if 'adv' in args.dataset:
        data = dataset.data
    else:
        data = dataset[0]
    data.all_features = all_features
    print('data feature', data.all_features)
    from model import SAGE, GCN, APPNP, MLP, SGC, GAT, IAPPNP, ORTGNN, LP
    from model_ALTOPT import ALTOPT, ExactALTOPT, AGD

    if args.model == 'SAGE':
        model = SAGE(in_channels=data.all_features,
                     hidden_channels=args.hidden_channels,
                     out_channels=dataset.num_classes, 
                     dropout=args.dropout,
                     num_layers=args.num_layers).cuda()

    elif args.model == 'GCN':
        model =  GCN(in_channels=data.all_features,
                     hidden_channels=args.hidden_channels,
                     out_channels=dataset.num_classes, 
                     dropout=args.dropout,
                     num_layers=args.num_layers, args=args).cuda()
        
    elif args.model == 'SGC':
        model = SGC(in_channels=data.all_features,
                    out_channels=dataset.num_classes, 
                    dropout=args.dropout).cuda()

    elif args.model == 'GAT':
        model = GAT(in_channels=data.all_features,
                    hidden_channels=8, 
                    num_layers=args.num_layers,
                    heads=8, 
                    out_channels=dataset.num_classes,
                    dropout=args.dropout).cuda()

    elif args.model == 'MLP':
        prop = Propagation(K=args.K,
                           alpha=args.alpha,
                           mode='APPNP',
                           cached=True,
                           args=args)

        model =  MLP(in_channels=data.all_features,
                     hidden_channels=args.hidden_channels,
                     out_channels=dataset.num_classes, 
                     dropout=args.dropout,
                     args=args,
                     prop=prop).cuda()

    elif args.model == 'APPNP':
        prop = Propagation(K=args.K,
                           alpha=args.alpha,
                           mode=args.prop,
                           cached=True,
                           args=args)

        model = APPNP(in_channels=data.all_features,
                      hidden_channels=args.hidden_channels, 
                      out_channels=dataset.num_classes, 
                      dropout=args.dropout,
                      num_layers=args.num_layers, 
                      prop=prop,
                      args=args).cuda()

    elif args.model == 'IAPPNP':
        prop = Propagation(K=args.K,
                           alpha=args.alpha,
                           mode=args.prop,
                           cached=True,
                           args=args)
        model = IAPPNP(in_channels=data.all_features,
                      hidden_channels=args.hidden_channels,
                      out_channels=dataset.num_classes,
                      dropout=args.dropout,
                      num_layers=args.num_layers,
                      prop=prop,
                      args=args).cuda()

    elif args.model == 'ORTGNN':
        model = ORTGNN(in_channels=data.all_features,
                      hidden_channels=args.hidden_channels,
                      out_channels=dataset.num_classes,
                      dropout=args.dropout,
                      num_layers=args.num_layers,
                      args=args).cuda()

    elif args.model == 'ALTOPT':
        prop =  Propagation(K=args.K, 
                    alpha=args.alpha, 
                    mode=args.prop,
                    cached=True,
                    args=args)
        
        model = ALTOPT(in_channels=data.all_features,
                       hidden_channels=args.hidden_channels, 
                       out_channels=dataset.num_classes, 
                       dropout=args.dropout, 
                       num_layers=args.num_layers, 
                       prop=prop,
                       args=args).cuda()
        
    elif args.model == 'AGD':
        prop =  Propagation(K=args.K, 
                    alpha=args.alpha, 
                    mode=args.prop,
                    cached=True,
                    args=args)
        
        model = AGD(in_channels=data.all_features,
                       hidden_channels=args.hidden_channels, 
                       out_channels=dataset.num_classes, 
                       dropout=args.dropout, 
                       num_layers=args.num_layers, 
                       prop=prop,
                       args=args).cuda()
    elif args.model == 'EXACT':
        prop =  Propagation(K=args.K, 
                    alpha=args.alpha, 
                    mode=args.prop,
                    cached=True,
                    args=args)
        
        model = ExactALTOPT(in_channels=data.all_features,
                       hidden_channels=args.hidden_channels, 
                       out_channels=dataset.num_classes, 
                       dropout=args.dropout, 
                       num_layers=args.num_layers, 
                       prop=prop,
                       args=args).cuda()
    elif args.model == 'CS':
        prop = Propagation(K=args.K,
                           alpha=args.alpha,
                           mode=args.prop,
                           cached=True,
                           args=args)
        model = IAPPNP(in_channels=data.all_features,
                       hidden_channels=args.hidden_channels,
                       out_channels=dataset.num_classes,
                       dropout=args.dropout,
                       num_layers=args.num_layers,
                       prop=prop,
                       args=args).cuda()
    elif args.model == 'LP':
        model = LP(args=args)

    else:
        raise ValueError('Model not supported')

    return model