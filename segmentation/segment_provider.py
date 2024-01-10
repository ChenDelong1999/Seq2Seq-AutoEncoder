from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import pickle
import base64
from segmentation import Segmenter

app = FastAPI()

class Input(BaseModel):
    content_lst: dict
    typ: str

class Response(BaseModel):
    result: dict

segmenter = Segmenter('mobile_sam_v2', '/home/dchenbs/workspace/cache/sam_weights/mobile_sam_v2/l2.pt')


@app.post("/segment_provider",response_model=Response)        
async def segment_provider(request: Input):
    time_start = time.time()

    masks = segmenter(request.content_lst['img_path'], post_processing=request.content_lst['post_processing'])
    
    # 将numpy数组转换为二进制数据，然后转换为字符串
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