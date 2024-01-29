from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import pickle
import base64
import pprint
from segmentation import Segmenter

app = FastAPI()

class Input(BaseModel):
    content_lst: dict

class Response(BaseModel):
    result: dict

# read args from segmnet_provider_config.json
with open('segment_provider_config.json', 'r') as f:
    config = json.load(f)
    pprint.pprint(config)

segmenter = Segmenter(config['model_name'], config['checkpoint'], **config['kwargs'])


@app.post("/segment_provider",response_model=Response)        
async def segment_provider(request: Input):
    time_start = time.time()

    masks = segmenter(request.content_lst['img_path'], post_processing=request.content_lst['post_processing'])
    
    masks_bytes = pickle.dumps(masks)
    masks_str = base64.b64encode(masks_bytes).decode('utf-8')

    res = {"response": masks_str}
    print(f'request received: {request}')
    print(f'request processed ({round(time.time()-time_start, 2)}s)')

    with open('segment_provider_log.txt', 'a') as f:
        f.write(json.dumps({
            'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'request': str(request),
            'time_elapsed': round(time.time()-time_start, 2)
        }, indent=4) + '\n')

    return Response(result=res)