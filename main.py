from os.path import exists, join
from timeit import default_timer
import torch
from torchvision import transforms, models
from torch import nn
import streamlit as st

# Decrease/Increase depend on the hardware 
batch_size = 150
device = 'cuda' if torch.cuda.is_available() else 'cpu'
database_root = './Landmark_Retrieval/Landmark_Retrieval/test/database_root/database'
query_root = './Landmark_Retrieval/Landmark_Retrieval/test/query_root/query'
class_names = [
    "bao_tang_ha_noi", "buu_dien_trung_tam_tphcm", "cau_long_bien", "cau_nhat_tan",
    "cau_rong", "cho_ben_thanh_tphcm", "chua_cau", "chua_mot_cot", "chua_thien_mu",
    "cot_co", "hoang_thanh", "hon_chong_nhatrang", "landmark81", "lang_bac",
    "lang_khai_dinh", "mui_ca_mau", "mui_ke_ga_phanthiet", "nha_hat_lon_hanoi",
    "nha_hat_lon_tphcm", "nha_tho_da_co_sapa", "nha_tho_lon_ha_noi", "quang_truong_lam_vien",
    "suoi_tien_tphcm", "thac_ban_gioc", "thap_cham", "thap_rua", "toa_nha_bitexco_tphcm",
    "tuong_chua_kito_vungtau", "ubnd_tphcm", "van_mieu"
]
@st.cache_resource
def load_models():
    """Load fine-tuned Resnet152"""
    pretrained_res_net = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    if device=="cuda":
      pretrained_res_net = pretrained_res_net.cuda()
    new_classifier = torch.nn.Sequential(*(list(pretrained_res_net.children())[:-1]))
    for param in pretrained_res_net.parameters():
        param.requires_grad = False
    pretrained_res_net.fc = nn.Sequential(
       nn.Linear(2048, 128),
       nn.ReLU(inplace=True),
       nn.Linear(128, 30),
    ).to(device)
    finetuned_check_point = torch.load("finetuned_res_net.pt")
    pretrained_res_net.load_state_dict(finetuned_check_point['model_state_dict'])
    return pretrained_res_net
    
model = load_models()

if not (exists("database.csv") and exists("query.csv")):
    import csv
    from pathlib import Path
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms, models
    from torchvision.datasets import ImageFolder

    # Define your data transformations
    start_time = default_timer()
    model = load_models()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    # Load test data using ImageFolder and apply data transformations
    database_data = ImageFolder(root="./Landmark_Retrieval/Landmark_Retrieval/test/database_root/", transform=transform)

    # Get the test images from the dataset
    test_loader = DataLoader(database_data, batch_size=batch_size)

    predicted_labels = []
    for images, _ in test_loader:
        predictions = model(images.to(device))
        predicted_labels.extend(predictions.argmax(dim=1).tolist())

    image_names = [Path(x[0]).name for x in database_data.samples]

    # Save image file names and predicted labels to a CSV file
    with open('database.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Label'])
        for image_name, label in zip(image_names, predicted_labels):
            writer.writerow([image_name, class_names[label]])
    
    query_data = ImageFolder(root="./Landmark_Retrieval/Landmark_Retrieval/test/query_root/", transform=transform)
    test_loader = DataLoader(query_data, batch_size=batch_size)
    predicted_labels = []

    for images, _ in test_loader:
        predictions = model(images.to(device))
        predicted_labels.extend(predictions.argmax(dim=1).tolist())

    image_names = [Path(x[0]).name for x in query_data.samples]

    with open('query.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Label'])
        for image_name, label in zip(image_names, predicted_labels):
            writer.writerow([image_name, class_names[label]])
    end_time = default_timer()
    st.write(f"initialize data takes {end_time-start_time} s")

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def compute_sift_features(image_path):
    """Compute SIFT features for an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors

def find_homography(query_keypoints, query_descriptors, result_keypoints, result_descriptors):
    """Find a homography between a query image and a result image using RANSAC."""
    # Create a BFMatcher object
    bf = cv2.BFMatcher()

    # Match the query and result descriptors
    matches = bf.knnMatch(query_descriptors, result_descriptors, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    if len(good_matches) < 4:
        return 0, []

    src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([result_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    num_inliers = np.sum(mask)

    return num_inliers, good_matches

def draw_matches(query_image_path, query_keypoints,
                 result_image_path, result_keypoints,
                 matches):
    
    """Draw the matches between a query image and a result image."""
    
    query_image = cv2.imread(query_image_path)
    result_image = cv2.imread(result_image_path)

    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    match_image = cv2.drawMatches(query_image, query_keypoints,
                                  result_image, result_keypoints,
                                  matches, None)

    return match_image


def rerank_and_output_to_text_file(
    database_features,
    query_features,
    result_images_dict,
):
    
    reranked_result_images_dict = {}
    
    with open("output.txt", 'w') as f:
        for query_image_name, result_image_names in result_images_dict.items():
            query_keypoints, query_descriptors = query_features[query_image_name]

            scores = []
            all_good_matches = []
            
            for result_image_name in result_image_names:
                result_keypoints, result_descriptors = database_features[result_image_name]

                num_inliers, good_matches = find_homography(query_keypoints,
                                                            query_descriptors,
                                                            result_keypoints,
                                                            result_descriptors)

                scores.append(num_inliers)
                all_good_matches.append(good_matches)

            reranked_result_image_names = [result_image_names[i] for i in np.argsort(scores)[::-1]]
            reranked_result_images_dict[query_image_name] = reranked_result_image_names
            query_id = query_image_name.replace('.jpg', '')
            result_ids = [result_image_name.replace('.jpg', '') for result_image_name in reranked_result_image_names[:10]]
            f.write(f"{query_id} {' '.join(result_ids)}\n")

database_df = pd.read_csv('database.csv')
query_df = pd.read_csv('query.csv')
result_df = pd.merge(database_df, query_df, on='Label')
grouped = result_df.groupby('Image_y')
result_images = grouped['Image_x'].apply(list)
result_images_dict = result_images.to_dict()

if st.button(
    "Run through all query image",
    help="Take all 300 query image, find top 10 result and output feature matching"
):

    database_root = './Landmark_Retrieval/Landmark_Retrieval/test/database_root/database'
    query_root = './Landmark_Retrieval/Landmark_Retrieval/test/query_root/query'

    database_features = {}
    for image_name in database_df['Image']:
        image_path = f'{database_root}/{image_name}'
        keypoints, descriptors = compute_sift_features(image_path)
        database_features[image_name] = (keypoints, descriptors)

    query_features = {}
    for image_name in query_df['Image']:
        image_path = f'{query_root}/{image_name}'
        keypoints, descriptors = compute_sift_features(image_path)
        query_features[image_name] = (keypoints, descriptors)

    reranked_result_images_dict = rerank_and_output_to_text_file(
        database_features,
        query_features,
        result_images_dict,
    )

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button(
    "Find similar image."
) and uploaded_file is not None:
    start_time_query = default_timer()
    query_image = Image.open(uploaded_file)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    processed_image = transform(query_image).to(device)
    processed_image = processed_image.unsqueeze(0) 
    model.eval()
    label = class_names[model(processed_image).argmax(dim=1).tolist()[0]]
    filtered_df = database_df.query('Label == @label')
    candidate_list = filtered_df["Image"].to_list()
    database_root = './Landmark_Retrieval/Landmark_Retrieval/test/database_root/database'
    st.image(query_image, caption='Uploaded Image.', use_column_width=True)
    gray = cv2.cvtColor(np.array(query_image), cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    query_keypoints, query_descriptors = sift.detectAndCompute(gray, None)
    num_inliers_list = []
    matches_list = []
    print(len(candidate_list))
    for image_name in candidate_list:
        result_image_path = join(database_root, image_name)
        result_keypoints, result_descriptors = compute_sift_features(result_image_path)

        num_inliers, good_matches = find_homography(query_keypoints, query_descriptors,
                                                    result_keypoints, result_descriptors)

        num_inliers_list.append(num_inliers)
        matches_list.append(good_matches)

    reranked_image_list, reranked_matches_list = list(zip(*sorted(zip(candidate_list, matches_list, num_inliers_list),
                                                             key=lambda x: x[2], reverse=True)))[:2]
    end_time_query = default_timer()
    st.write(f"Query take: {end_time_query - start_time_query}s")
    start_time_display = default_timer()
    match_images = []
    for image_name, matches in zip(reranked_image_list, reranked_matches_list):
        result_image_path = join(database_root, image_name)
        result_keypoints, _ = compute_sift_features(result_image_path)
        result_image = cv2.imread(result_image_path)

        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        match_image = cv2.drawMatches(np.array(query_image), query_keypoints,
                                      result_image, result_keypoints,
                                     matches, None)
        match_images.append(match_image)
    i = 0
    for m_image in match_images:
        if i >= 10:
            break
        st.image(m_image, caption=f"top {i+1}: {reranked_image_list[i]}"    )
        i += 1
    end_time_display = default_timer()
    st.write(f"Draw matches takes {end_time_display-start_time_display}s")