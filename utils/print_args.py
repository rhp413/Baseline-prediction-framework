def print_args(args):
    """Print experiment arguments in a simple format"""
    print('Args in experiment:')
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()
