def zip():
    from zipfile import ZipFile
    import os
    import zipfile
    path_model = '../models'
    name = 'models0'
    old_path = os.getcwd()
    os.chdir(f"{path_model}/")
    with zipfile.ZipFile(f'/srv/data/apatshin_docker/data/input/{name}.zip', 'w') as myzip:
        for file in os.listdir():
            if 'pth' in file:
                myzip.write(file)

    os.chdir(f"{old_path}")


if __name__ == '__main__':
    zip()