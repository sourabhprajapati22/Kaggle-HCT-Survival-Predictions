from pathlib import Path
import os

project_name='HCT-Survival'

list_of_files=[
    f'{project_name}/data/raw/',
    f'{project_name}/data/processed/',
    f'{project_name}/notebooks/eda.ipynb',
    f'{project_name}/notebooks/feature_engineering.ipynb',
    f'{project_name}/notebooks/model_training.ipynb',
    f'{project_name}/notebooks/evaluation.ipynb',
    f'{project_name}/src/data_processig.py',
    f'{project_name}/src/train.py',
    f'{project_name}/src/infrence.py',
    f'{project_name}/models/',
    f'{project_name}/scripts/run_train.sh',
    f'{project_name}/logs',
    f'{project_name}/reports/',
    f'{project_name}/configs/',
    'requirements.txt',
]

for file in list_of_files:
    file_path=Path(file)
    file_dir,file_name=os.path.split(file_path)
    print(file_dir,file_name)
    if (not os.path.exists(file_path)) and ('.' in str(file_path)):
        if len(file_dir) !=0:
            os.makedirs(file_dir,exist_ok=True)
            if len(file_name)!=0:
                with open(file_path,'w') as f:
                    print(f'1.create dir {file_dir} with name {file_name}')

        else:
            with open(file_path,'w') as f:
                print(f'2.create {file_name}')
    
    else:
        os.makedirs(file_path,exist_ok=True)
        print(f'3. crate dir {file_dir}')