# Text to Image Manipulation
### project using CLIP (Multimodal Learning)   
- CLIP (Contrastive Language-Image Pre-Training) [[paper](https://arxiv.org/pdf/2103.00020.pdf)][[code](https://github.com/openai/CLIP)]

### Training Mapper  
#### Method 1. using text embedding vector obtained from torch.nn.embedding   
    
![그림1](https://user-images.githubusercontent.com/74402562/130311269-b62aac08-40e8-4218-b48e-fc8e6a09eac4.png)   


#### Method 2. using text embedding vector obtained from CLIP's text encoder   

![clip_flow](https://user-images.githubusercontent.com/74402562/130311270-0cab934b-4383-4ffc-9665-9bf11b1923b5.png)

#### Mapper Network Structures
- vector concatenate method   

    <img src="https://user-images.githubusercontent.com/74402562/130311813-ddbd04fc-a75a-4b01-ba0b-ee42aaf67c39.png" width="40%" height="40%"></img>    
    
- vector summation method    

    <img src="https://user-images.githubusercontent.com/74402562/130311815-705aca5c-61ce-4e64-9839-37c2386309a7.png" width="40%" height="40%"></img>
    
#### Example of Inference Process
- Single module    
 
    <img src="https://user-images.githubusercontent.com/74402562/130312144-8faee752-ab9c-40d1-a563-f570fb94623b.png" width="80%" height="80%"></img>

- Multi modules combination    
 
    <img src="https://user-images.githubusercontent.com/74402562/130312138-a3bbd4ef-8f39-4a53-9270-44d7680f06ed.png" width="100%" height="100%"></img>
