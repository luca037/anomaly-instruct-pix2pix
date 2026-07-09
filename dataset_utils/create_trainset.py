import json

JSON_INPUT = "./mirage_hz_pill.json"
MVTEC_PATH = "~/big_disk/datasets/mvtec/"
MYDATASET_PATH = "~/big_disk/datasets/my_dataset/"


def cp_mvtec():
    import os

    with open(JSON_INPUT, "r") as f:
        data = json.load(f)

    for _, dic in data.items():
        input_paths =  [MVTEC_PATH + path for path in dic['base_img_path']]
        output_paths = [MYDATASET_PATH + original_name(path) for path in dic['base_img_path']]
        for input, output in zip(input_paths, output_paths):
            command = f"cp {input} {output}"
            print("Executing:", command)
            status = os.system(command)
            if status:
                print("ERROR with command:", command)


def original_name(path):
    splitted = path.split('/')
    out = splitted[0] + '_' + splitted[-2] + '_' + splitted[-1]
    return out

def edit_name(path):
    splitted = path.split('/')
    out = splitted[0] + '_' + splitted[-1]
    return out

def create_metadata():
    with open(JSON_INPUT, "r") as f:
        data = json.load(f)

    entries = []
    for _, dic in data.items():
        original_imgs = [original_name(path) for path in dic['base_img_path']]
        edited_imgs = [edit_name(path) for path in dic['img_path']]
        edit_prompts = [prompt for prompt in dic['defect_synonym']]

        for o, e, p in zip(original_imgs, edited_imgs, edit_prompts):
            entry = {
                'original_image': o,
                'edited_image': e,
                'edit_prompt': p
            }
            entries.append(entry)
    
    with open("metadata.jsonl", 'w') as f:
        for entry in entries:
                json.dump(entry, f)
                f.write('\n')


def main():
    create_metadata()
    #cp_mvtec()


if __name__ == '__main__':
    main()