import medmnist
from medmnist import INFO

print('Dataset classes:')
for dataset in ['octmnist', 'tissuemnist', 'pathmnist']:
    num_classes = len(INFO[dataset]['label'])
    print(f'{dataset}: {num_classes} classes')
    print(f'  Labels: {INFO[dataset]["label"]}')

print('\nHAM10000: 7 classes')
print('  Labels: ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]')