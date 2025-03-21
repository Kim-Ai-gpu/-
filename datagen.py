import os
import random
import torch
import torchvision.transforms as T
import re
from combineimage import Combine

def extract_number_from_filename(filename):
    match = re.match(r'(\d+)', filename)
    if match:
        return match.group(1)
    else:
        return None

def pick_random_files(folder_path, n):
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if len(files) < n:
            print(f"경고: {n}개의 파일을 찾을 수 없습니다. 파일 개수: {len(files)}")
            return files
        selected_files = random.sample(files, n)
        return selected_files
    except Exception as e:
        print(f"에러 발생: {e}")
        return []


input_dir = './dataset/inputs'
label_dir = './dataset/labels'
os.makedirs(input_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

folder_path = './sp'  
combiner = Combine()

num_samples = 10000 
for i in range(num_samples):
    n = random.randint(1, 9)
    if n % 2 != 0:
        n += 1
    

    selected_files = pick_random_files(folder_path, n)
    if not selected_files:
        continue
    
    combined_tensor = combiner.combine(selected_files)
    if combined_tensor is None:
        continue


    combined_img = T.ToPILImage()(combined_tensor)
    
    label_tensor = torch.zeros(1, 99,dtype=torch.int8)
    element_numbers = []
    for file in selected_files:
        num_str = extract_number_from_filename(file)
        if num_str is not None:
            index = int(num_str) - 1  
            if 0 <= index < 99:
                label_tensor[0, index] = 1
                element_numbers.append(num_str)
    
    input_filename = os.path.join(input_dir, f'input_{i}.png')
    combined_img.save(input_filename)
    label_filename = os.path.join(label_dir, f'label_{i}.pt')
    torch.save(label_tensor, label_filename)
    
    print(f"Sample {i}: 저장 완료. 선택된 원소 번호: {element_numbers}")
