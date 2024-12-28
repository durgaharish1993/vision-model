
from typing import List, Dict, Optional, Union, Tuple, Iterable
import torch 
from PIL import Image 
import numpy as np 


def resize( image : Image, size   : Tuple[int, int], resample : Image.Resampling = None, reducing_gap : Optional[int] = None, ) -> np.ndarray : 
    height, width = size 
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap = reducing_gap
    )

    return resized_image

def rescale(image : np.ndarray, scale : float, dtype : np.dtype = np.float32 ):
    rescaled_image = image * scale 
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(image : np.ndarray, mean : Union[float,Iterable[float]], std : Union[float, Iterable[float]]):
    mean = np.array(mean, dtype=image.dtype)
    std  = np.array(std, dtype=image.dtype)
    image = (image-mean)/std 
    return image 

def process_image( images : List[Image.Image], size : Dict[str,int] = None,  resample : Image.Resampling = None, rescale_factor : float = None, image_mean : Optional[Union[float, List[float]]] = None, image_std  : Optional[Union[float, List[float]]] = None):
    height, width = size[0], size[1]
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]
    images = [normalize(image, mean=image_mean, std = image_std) for image in images]
    images = [image.transpose(2,0,1) for image in images]
    return images 


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token ):
    str_value = f"{image_token * image_seq_len} {bos_token} {prefix_prompt}"
    return str_value


class PaliGemmaProcessor: 
    def __init__(self,tokenizor,  
                 num_image_tokens : int,
                 image_size : int):
        
        self.image_seq_length = num_image_tokens
        self.image_size       = image_size 

        self.IMAGE_TOKEN = "<image>"

        tokens_to_add = {"additional_special_tokens" : [self.IMAGE_TOKEN]}
        tokenizor.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizor.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizor.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizor.add_bos_token = False
        tokenizor.add_eos_token = False

        self.tokenizer = tokenizor

    
    
    def __call__(self, text : List[str],
                 images : List[Image.Image],
                 padding : str = "longest",
                 truncation : bool = True):
        
        process_image(
            images, 
            size =(self.image_size, self.image_size) 
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std  = IMAGENET_STANDARD_STD,
        )
        
        pixel_values = np.stack(pixel_values, axis=0 )
        pixel_values = torch.tensor(pixel_values)

        input_strings = [ 
            add_image_tokens_to_prompt(
                prefix_prompt = prompt, 
                bos_token = self.tokenizer.bos_token, 
                image_seq_len = self.image_seq_length, 
                image_token = self.IMAGE_TOKEN
            ) for prompt in text
        ]

        inputs  = self.tokenizer(
                    input_strings, 
                    return_tensors = "pt",
                    padding = padding, 
                    truncation = truncation
        )

        return_data = {"pixel_values" : pixel_values, **inputs}

        return return_data


        



