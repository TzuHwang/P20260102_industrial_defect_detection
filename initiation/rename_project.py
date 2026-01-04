import os


def rename_project(old_name, new_name):
    # 1. Replace strings inside files
    for root, dirs, files in os.walk('.', topdown=False):
        # Skip the .git directory to avoid corrupting git history
        if '.git' in root:
            continue

        for filename in files:
            # Skip the script itself
            if filename == 'rename_project.py':
                continue

            file_path = os.path.join(root, filename)

            # Read and replace content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if old_name in content:
                    new_content = content.replace(old_name, new_name)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f'Updated content: {file_path}')
            except (UnicodeDecodeError, PermissionError):
                # Skip binary files or protected files
                continue

        # 2. Rename files and directories
        for name in files + dirs:
            if old_name in name:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, name.replace(old_name, new_name))
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} -> {new_path}')


if __name__ == '__main__':
    target_name = input('Enter the new project name: ')
    rename_project('private_projet_template', target_name)
    print('\nDone! All instances replaced.')
