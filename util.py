def get_task_names(data_dirs):
    task_names = []
    for data_dir in data_dirs:
        file_name = data_dir.split('/')[-1]
        task_name = file_name.split('.')[0]
        task_names.append(task_name)
    return task_names