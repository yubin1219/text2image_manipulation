# Text to Image Manipulation
## Multi modal learning project
- using OpenAI CLIP (Contrastive Language-Image Pre-Training) [[paper](https://arxiv.org/abs/2103.00020)] [[code](https://github.com/openai/CLIP)]

## Latent Optimization

- G(w)를 CLIP의 image encoder에 넣어 임베딩 값을 구하고 CLIP의 text encoder에서 임베딩된 값과 코사인 유사도를 계산하여 loss로 추가 사용
- G(w) 임베딩 값이 text 임베딩 값과 유사해지는 방향으로 gradient decent를 이용해 latent vector w를 update
- text 의미에 맞게 이미지 manipulation은 잘 하지만 매번 몇 분의 optimization과정이 필요

## Global Directions
- text prompt를 style space S의 single, global direction과 mapping하는 방법
- Text prompt가 주어지면 CLIP을 통해 original text와 target text의 임베딩 차이인 delta t를 구한 후 기존 style과 변경된 style의 차이인 delta s에 mapping
- 수용할 수 있는 text의 범위가 넓음

## Latent Mapper 
**" Main Method "**

- Mapper는 latent vector w를 text의 의미에 맞게 변화시키는 방향으로 Training
- Training Process
   1. Encoder를 통해 original image의 latent vector를 얻는다.
   2. Encoder를 통해 얻은 latent vector는 text embedding vector와 summaiton 또는 concatenate되어 Mapper의 input으로 들어간다.
   3. Mapper를 통해 얻은 latent vector와 기존 latent vector w는 summation되어 StyleGAN2의 input으로 들어간다.
   4. StyleGAN2에서 생성한 이미지 G(w_hat)과 text 간의 유사도(Similarity Loss)를 CLIP model을 통해 구한다.
   5. Mapper는 G(w_hat)과 text 간의 유사도를 최소화시키는 latent vector를 뽑아내도록 학습된다.

- Advantages
   - Mapper를 Training 시킴으로써 매번 optimization과정을 거쳐야 하는 Latent Optimization 방식의 단점을 보완
   - text embedding vector를 사용하여 기존 Mapper의 단점을 개선
   - 하나의 모델에서 multi style을 learning 하는 효율적인 방법
- 단, 학습되어 있지 않은 style은 수행하지 못함

### Method 1. Use text embedding vector obtained from torch.nn.embedding

<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/130408521-54a5b4ad-a7e5-4f09-837c-febddb345066.png" width="100%" height="100%">
</p>
<br/>

### Method 2. Use text embedding vector obtained from CLIP text encoder

<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/130408531-b6e9218f-5b57-4396-b99f-19b673b823f6.png" width="100%" height="100%">
</p>
<br/>

### Method 3. Use Single Model for Multi Style Combination

<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/130411102-f1b3fb01-4033-48ce-8bb0-bdec59385918.png" width="100%" height="100%">
</p>
<br/>

- 이전 method들은 하나의 모델에서 multi style을 학습하기는 하지만 그 style들이 결합된 상태는 학습하지 못함
- 이 method는 하나의 모델에서 각각의 style과 그 style들이 결합된 상태까지 Training
- Inference 속도 단축

# Setup
- install CLIP
```
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
```

# Usage
## Train    
### Setup
#### Download pretrained model

- Pretrained Stylegan2

   ```
   wget https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EVv6yusEt1tFhrL3TCu0Ta4BlpzW3eBMTS0yTPKodNHsNA?download=1 -O stylegan2-ffhq-config-f.pt
   ```
   Or download from here : [click](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)

- pretrained model for identity loss 

   model_ir_se50 : [click](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view)

####  Guidance  

- Main training script is placed in ```mapper/train.py```
- Training arguments can be found at ```mapper/options/train_options.py```
- To add or modify text to learn, modify it in ```mapper/dataset/latents_dataset.py```
- If you perform an edit that is not supposed to change "colors" in the image, it is recommended to use the flag ```--no_fine_mapper```
    * To learn hair style text, use the flag ```--no_fine_mapper``` so that the hair color does not change
- To resume a training, please provide ```--checkpoint_path```


### Method 1. Use text embedding vector obtained from torch.nn.embedding

#### Vector Concatenate    

Example for training a mapper for hairstyle:

```
cd mapper
python train.py --data_mode "hair" --train_dataset_size 25000 --mapper_mode "Mapper_cat" --no_fine_mapper
```

#### Vector Summation

Example for training a mapper for female celeb:

```
cd mapper
python train.py --data_mode "female" --train_dataset_size 25000 --mapper_mode "Mapper_sum"
```
<br/>

### Method 2. Use text embedding vector obtained from CLIP text encoder

Example for training a mapper for color:

```
cd mapper
python train.py --data_mode "color" --text_embed_mode "clip_encoder" --mapper_mode "Mapper_cat"
```
<br/>

### Method 3. Use Single Model for Multi Style Combination

Example for training a mapper for multi style:

```
cd mapper
python train.py --data_mode "multi" --train_dataset_size 78000 --mapper_mode "Mapper_multi"
```
<br/>

## Inference     
#### Guidance
- Google drive addresses of pretrained weights and dataset download function are placed in ```mapper/utils.py```
- If ```--weights_download``` is set to True, the dataset and weights required in the inference process are automatically downloaded.
- Inference argument is placed in ```mapper/options/test_options.py```
- ```--num``` means which late vector to use. ```default = 60 ```
- ```--modules```and ```--texts``` must be paired.
- Images generated during inference are stored in the form of ```result_[input texts].png```
- ```--new_latent```를 써주면 원하는 이미지를 latent vector로 바꿔 사용할 수 있습니다. ```default = False```
   * Enter image file path in ```--new_image_path```. ex) "ubin.jpg"
   * ```"e4e_ffhq_encode.pt"``` and ```"shape_predictor_68_face_landmarks.dat"``` files required during the encoder process can be automatically downloaded from the code. If it's impossible, you can download it in the Google drive folder [[styleclip](https://drive.google.com/drive/folders/1kWkwihhYAg6mLffcxHzFofucM1dkVKVs?usp=sharing)]

### 1. Multi Model Combination    

<div align="center">
    <img src="https://user-images.githubusercontent.com/74402562/130411804-6bdc90ae-9a79-48aa-b53a-7b77224b4b8d.png" width="100%" height="100%">
</div>
<br/>
<br/>

modules|texts
-------|-----
celeb_female|["Elsa", "Anna", "Emma_Stone", "Anne_Hathaway", "Scarlett_Johansson"]
celeb_male|["Ironman", "Dicaprio", "Zuckerberg", "Tom_Holland"]
hair_sum or hair_cat |["curly", "wavy", "long", "bobcut" , "bangs"]
color_sum or hair_cat |["blonde", "pink", "blue", "black"]  
Disney_clip|["Elsa", "Anna", "Rapunzel", "Ariel"]
hair_clip |["wavy", "long", "bangs", "bobcut"]
color_clip|["blonde", "red", "pink", "blue", "purple", "brown", "black"]

Example for testing a mapper for 3 modules combination:

```
cd mapper
python inference.py --latent_path "test_female.pt" --modules celeb_female --modules hair_sum --modules color_sum --texts Emma_Stone --texts wavy --texts pink
```
<br/>

### 2. Single Model Learned Multi-Style Combination    

<div align="center">
    <img src="https://user-images.githubusercontent.com/74402562/130412529-82878ab4-70cb-458a-b31c-7318a29ae5d0.png" width="100%" height="100%">
</div>
<br/><br/>

texts|
-----|
["Elsa" , "Emma_Watson", "wavy", "bangs", "pink", "black"]|

Example for testing a mapper for single model learned multi-style:
```
cd mapper
python inference_single.py --latent_path "test_female.pt" --texts Emma_Watson --texts wavy --texts black
```
<br/>

# Dataset
- CelebA-HQ images are converted into latent vectors through encoder4editing and used as a dataset.    
- Through the CLIP model, each image of the train set and test set is classified into female or male and used.    


Dataset | Description
--------|-------------
[train set](https://drive.google.com/file/d/1gof8kYc_gDLUT4wQlmUdAtPnQIlCO26q/view)| 총 24179개의 latent vector
[test set](https://drive.google.com/file/d/1j7RIfmrCoisxx3t-r-KC02Qc8barBecr/view)| 총 2824개의 latent vector
[train_female](https://drive.google.com/file/d/1_ZTJa9VmhWaU5xyzyAdK4GHxhnzaEORx/view?usp=sharing)|총 15332개의 latent vector
[train_male](https://drive.google.com/file/d/1u9r3qcH7qqGHGoolkaR-xEyNCDIz0bCv/view?usp=sharing)|총 8844개의 latent vector
[test_female](https://drive.google.com/file/d/1LYOdh-45aNbGwaj38qQObbBX73wMJXu1/view?usp=sharing)|총 1899개의 latent vector
[test_male](https://drive.google.com/file/d/11MGereOsRMUo_HQXV8NnFR-L_ECcTXvk/view?usp=sharing)|총 925개의 latent vector
<br/>

# Using t-SNE for vector visualization
### **Each modules - celeb / hair style / color**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130887843-74b37115-9b4a-4a7a-84c2-2ded48594318.png" width="30%" height="30%"> <img src="https://user-images.githubusercontent.com/74402562/130888094-e8202de7-8ed0-4aa2-9609-8856c4c19ee6.png" width="30%" height="30%"> <img src="https://user-images.githubusercontent.com/74402562/130886760-7a92af0b-dfd7-4954-a7e4-e91838400ce0.png" width="30%" height="30%"></p>

### **Single model learned multi-style combination**    
<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130884002-e9832c50-5d20-4e5f-93d7-7545e5344529.png" width="100%" height="100%"></p>
<br/>

# Examples of Result
### Latent Optimization
**<p align="center">Input Image / "He is a young boy" / "He is smiling"</p>**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130747473-c0e6d2bb-690e-4a64-84a9-a898ac10c874.png">
<img src="https://user-images.githubusercontent.com/74402562/130758910-09c68766-8f4a-4cec-ab97-0a9219bc0ffe.gif"> <img src="https://user-images.githubusercontent.com/74402562/130758900-cf0c1b04-babd-464e-a79c-3f12ea469be6.gif"></p>

### Latent Mapper    
- **Results of Changed Hair Style**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130761194-b55a756e-a519-4212-ae23-83e717743dbb.png"></p>

- **Results of Changed Hair Color**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130775347-c4bf89ef-06b1-4315-9c3d-664fc81102df.png" width="80%" height="80%"></p>

- **Results of Male celeb style**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130893205-8c761c0a-f1f9-4548-8ed4-9e3975bc3293.png"></p>

- **Results of Multi-Styling**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130781810-8bab03e5-e887-4a0d-b7fc-9bdb894034c7.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130781822-27aef3cd-72df-4f86-9311-bc932a93dc63.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130781826-a124a7e3-fda9-4871-8174-449687853fed.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130781834-2236dbf6-ee23-4ea0-bfe4-3cf5178d9f0c.png"></p>
