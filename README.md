# 2021 Summer Intern

# Project Description     
text를 사용해 image manipulation을 수행하는 project입니다.    
Pretrained model인 StyleGAN2와 CLIP을 활용한 다음과 같은 방식들을 통해 text to image manipulation을 수행하였습니다.

## Latent Optimization [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13V5c4sTGwYHEsLuxEkVTBApKPCHR5-g9?usp=sharing)

- G(w)를 CLIP의 image encoder에 넣어 임베딩 값을 구하고 CLIP의 text encoder에서 임베딩된 값과 코사인 유사도를 계산하여 loss로 추가 사용
- G(w) 임베딩 값이 text 임베딩 값과 유사해지는 방향으로 gradient decent를 이용해 latent vector w를 update
- 이미지 조작은 잘 하지만 매번 몇 분의 optimization과정이 필요

## Global Directions [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QtDtvAZvS7lDXf2_gPbDi6L4nLWcK5WY?usp=sharing)
- text prompt를 style space S의 single, global direction과 mapping하는 방법
- Text prompt가 주어지면 CLIP을 통해 original text와 target text의 임베딩 차이인 delta t를 구한 후 기존 style과 변경된 style의 차이인 delta s에 mapping
- 수용할 수 있는 text의 범위가 넓음

## Latent Mapper [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15cdrsZPId89EiImjVU0LzLH7a5h3AeWt?usp=sharing)   
**이번 프로젝트의 "Main Method"**

- Mapper는 latent vector w를 text의 의미에 맞게 변화시키는 방향으로 Training
- Training 과정
   1. 이미지 정보가 담긴 latent vector는 text embedding vector와 summaiton 또는 concatenate되어 Mapper의 input으로 들어간다.
   2. Mapper를 통해 얻은 latent vector와 기존 latent vector w는 summation되어 StyleGAN2의 input으로 들어간다.
   3. StyleGAN2에서 생성한 이미지 G(w_hat)과 text 간의 유사도(Similarity Loss)를 CLIP model을 통해 구한다.
   4. Mapper는 G(w_hat)과 text 간의 유사도를 최소화시키는 latent vector를 뽑아내도록 학습된다.

- Mapper를 Training 시킴으로써 매번 optimization과정을 거쳐야 하는 Latent Optimization 방식의 단점을 보완
- text embedding vector를 사용하여 기존 Mapper의 단점을 개선
- 하나의 모델에서 multi style을 learning 하는 효율적인 방법
- 단, 학습되어 있지 않은 style은 수행하지 못함

### Method 1. Use text embedding vector obtained from torch.nn.embedding

<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/130408521-54a5b4ad-a7e5-4f09-837c-febddb345066.png" width="90%" height="90%">
</p>
<br/><br/>

- nn.embedding을 통해 얻은 text embedding vector를 latent vector와 summation 또는 concatenate하여 Mapper의 input으로 사용
<br/>

### Method 2. Use text embedding vector obtained from CLIP text encoder

<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/130408531-b6e9218f-5b57-4396-b99f-19b673b823f6.png" width="90%" height="90%">
</p>
<br/><br/>

- CLIP의 text encoder를 통해 얻은 text embedding vector를 latent vector와 concatenate하여 Mapper의 input으로 사용
<br/>

### Method 3. Use Single Model for Multi Style Combination

<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/130411102-f1b3fb01-4033-48ce-8bb0-bdec59385918.png" width="90%" height="90%">
</p>
<br/><br/>

- 이전 method들은 하나의 모델에서 multi style을 학습하기는 하지만 그 style들이 결합된 상태는 학습하지 못함
- 이 method는 하나의 모델에서 각각의 style과 그 style들이 결합된 상태까지 Training

# Setup
- install CLIP
```
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
```

# Usage
## Train [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15cdrsZPId89EiImjVU0LzLH7a5h3AeWt?usp=sharing)    
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

- Main training script는 ```mapper/train.py```입니다.
- Training arguments는 ```mapper/options/train_options.py```에 있습니다.
- Namespace를 사용하려면 ```mapper/train.py``` 에서 주석 처리된 코드를 살려주면 됩니다. 기존에는 ```ArgumentParser```를 사용한 코드로 적용되어 있습니다.
- 학습할 text를 추가하거나 수정하려면 ```mapper/dataset/latents_dataset.py```에서 수정하면 됩니다.
- Original Image에서 'color'는 변화시키고 싶지 않다면, ```no_fine_mapper = True```로 설정해야 합니다.
    * hair style 변화를 학습시키려면 ```no_fine_mapper=True```로 설정해야 머리 색상이 바뀌지 않습니다.
- weight를 불러와서 학습을 이어가려면, ```--checkpoint_path```에 weight명을 설정해주면 됩니다.


### Method 1. Use text embedding vector obtained from torch.nn.embedding

#### Vector Concatenate    

Example for training a mapper for hairstyle:

```
cd mapper
python train.py --data_mode "hair" --train_dataset_size 25000 --mapper_mode "Mapper_cat" --no_fine_mapper
```

위 코드 실행 방식은 ```ArgumentParser```를 사용할 때의 방식입니다.    
```Namespace```를 사용할  ```opts.data_mode = "hair" , opts.mapper_mode = "Mapper_cat" , opts.no_fine_mapper = True```로 설정해주면 됩니다.

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

## Inference [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15cdrsZPId89EiImjVU0LzLH7a5h3AeWt?usp=sharing)    
#### Guidance
- ```mapper/utils.py```에 데이터셋과 pretrained weights의 google drive 주소와 download 함수가 있습니다.
- ```--weights_download```를 True로 하면 inference과정에서 필요한 dataset과 weight들이 자동으로 다운로드 됩니다.
- inference argument는 ```mapper/options/test_options.py```에 있습니다.
- ```Namespace```를 사용하려면 ```mapper/inference.py``` 에서 주석 처리된 코드를 살려주면 됩니다. 기존에는 ```ArgumentParser```를 사용한 코드로 적용되어 있습니다.
- ```--w_num```은 몇 번째 latent vector를 사용할 것인지를 의미합니다. 숫자를 입력하여 원본 이미지를 바꿀 수 있습니다. ```default = 60 ```
- ```--modules```는 사용할 module을 하나씩 입력하면 list에 담아집니다.
- ```--texts```는 style을 하나씩 입력하면 list에 담아집니다.
- ```--modules```와 ```--texts```는 짝이 맞아야 합니다.
- inference 시 생성된 images는 ```result_[입력한 texts].png``` 형태로 저장됩니다.

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
python single_inference.py --latent_path "test_female.pt" --texts Emma_Watson --texts wavy --texts black
```

# Dataset
- CelebA-HQ 이미지를 encoder4editing을 통해 latent vector로 변환시켜 데이터셋으로 사용했습니다.   
- CLIP model을 통해 train set과 test set을 female과 male로 분류하여 사용했습니다.


Dataset | Description
--------|-------------
[train set](https://drive.google.com/file/d/1gof8kYc_gDLUT4wQlmUdAtPnQIlCO26q/view)| 총 24179개의 latent vector
[test set](https://drive.google.com/file/d/1j7RIfmrCoisxx3t-r-KC02Qc8barBecr/view)| 총 2824개의 latent vector
[train_female](https://drive.google.com/file/d/1_ZTJa9VmhWaU5xyzyAdK4GHxhnzaEORx/view?usp=sharing)|총 15332개의 latent vector
[train_male](https://drive.google.com/file/d/1u9r3qcH7qqGHGoolkaR-xEyNCDIz0bCv/view?usp=sharing)|총 8844개의 latent vector
[test_female](https://drive.google.com/file/d/1LYOdh-45aNbGwaj38qQObbBX73wMJXu1/view?usp=sharing)|총 1899개의 latent vector
[test_male](https://drive.google.com/file/d/11MGereOsRMUo_HQXV8NnFR-L_ECcTXvk/view?usp=sharing)|총 925개의 latent vector

# Examples of Result
### Latent Optimization
**<p align="center">Input Image / "He is a young boy" / "He is smiling"</p>**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130747473-c0e6d2bb-690e-4a64-84a9-a898ac10c874.png">
<img src="https://user-images.githubusercontent.com/74402562/130758910-09c68766-8f4a-4cec-ab97-0a9219bc0ffe.gif"> <img src="https://user-images.githubusercontent.com/74402562/130758900-cf0c1b04-babd-464e-a79c-3f12ea469be6.gif"></p>

### Latent Mapper    
- **Results of Changed Hair Style**
<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130761194-b55a756e-a519-4212-ae23-83e717743dbb.png"></p>

- **Reults of Changed Hair Color**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130775347-c4bf89ef-06b1-4315-9c3d-664fc81102df.png" width="80%" height="80%"></p>

- **Reults of Multi-Styling**

<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130781810-8bab03e5-e887-4a0d-b7fc-9bdb894034c7.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130781822-27aef3cd-72df-4f86-9311-bc932a93dc63.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130781826-a124a7e3-fda9-4871-8174-449687853fed.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/74402562/130781834-2236dbf6-ee23-4ea0-bfe4-3cf5178d9f0c.png"></p>

