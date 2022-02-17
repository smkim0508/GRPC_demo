import os
import argparse

import grpc
import proto_sample_pb2, proto_sample_pb2_grpc

import cv2
import numpy as np


def request(ip, port, frame):
    if ip[-1] != ':':
        ip += ':'

    with grpc.insecure_channel(ip+port) as channel:
        stub = proto_sample_pb2_grpc.AI_ModelServiceStub(channel)
        response = stub.process(
            proto_sample_pb2.ClientRequest(
                img_bytes=bytes(frame),
                width=frame.shape[1],
                height=frame.shape[0],
                channel=frame.shape[2]
            ))
        return response

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='50051', help='port number (default: 50051)')
    parser.add_argument('--ip', type=str, default='localhost:', help='ip address (default: localhost:')
    return parser.parse_args()

def _get_extension(img_name):
    return img_name.split('.')[-1]

def _recover_image_from_bytestream(bytes, height, width, channel=3):
    image = np.array(list(bytes))
    image = image.reshape((height, width, channel))
    image = np.array(image, dtype=np.uint8)

    return image

def main():
    args = opt()
    
    image_list = ['demo.jpeg']
    
    for idx, file_path in enumerate(image_list):
        img_ext = _get_extension(file_path)
        if 'jpeg' != img_ext:
            raise NotImplementedError('Image decoding type must be jpeg but got {}'.format(img_ext))
            continue
        
        src = cv2.imread(file_path)
        height, width, channel = src.shape


        # inference
        response = request(args.ip, args.port, src)
        
        # recover to np.ndarray
        dst = _recover_image_from_bytestream(response.img_bytes, response.height, response.width, response.channel)
        
        # write result file
        out_image_path = 'result-{}.jpeg'.format(idx)
        cv2.imwrite(out_image_path, dst)
        print(out_image_path)

if __name__ == '__main__':
    main()