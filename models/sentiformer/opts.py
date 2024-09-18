import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',        
                 type=str,
                 default='FI',
                 help='FI, Twitter_LDL or Artphoto'),
            dict(name='--dataPath',
                 default="./datasets/processed_data_clip.pkl",
                 type=str,
                 help=' '),
            dict(name='--seq_lens',     
                 default=[50, 50, 50],
                 type=list,
                 help=' '),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
           dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
            dict(name='--test_checkpoint',
                 default="./checkpoint/test/Artphoto_Acc7_Best.pth",
                 type=str,
                 help=' '),
        ],
        'network': [
            dict(name='--CUDA_VISIBLE_DEVICES',        
                 default='0',
                 type=str),
            dict(name='--fusion_layer_depth',
                 default=6,
                 type=int)
        ],

        'common': [
            dict(name='--project_name',    
                 default='MET_Demo',
                 type=str
                 ),
           dict(name='--is_test',    
                 default=1,
                 type=int
                 ),
            dict(name='--seed',  # try different seeds
                 default=0,
                 type=int
                 ),
            dict(name='--models_save_root',
                 default='./checkpoint',
                 type=str
                 ),
            dict(name='--batch_size',
                 default=64,
                 type=int,
                 help=' '),
            dict(
                name='--n_threads',
                default=1,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(name='--lr',
                 type=float,
                 default=1e-4),
            dict(name='--weight_decay',
                 type=float,
                 default=1e-4),
            dict(
                name='--n_epochs',
                default=100,
                type=int,
                help='Number of total epochs to run',
            )
        ]
    }

    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()
    return args