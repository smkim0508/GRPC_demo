import grpc
import proto_sample_pb2, proto_sample_pb2_grpc
from concurrent import futures

import os
import cv2
import numpy as np
import argparse

from datetime import datetime
from pytz import timezone
import logging


def _recover_image_from_bytestream(bytes, height, width, channel=3):
    image = np.array(list(bytes))
    image = image.reshape((height, width, channel))
    image = np.array(image, dtype=np.uint8)

    return image


class MyAI_Model(proto_sample_pb2_grpc.AI_ModelService):
    def __init__(self, args):
        super(MyAI_Model, self).__init__()

        # logging
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
            
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.stream_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(os.path.join(args.logdir, __class__.__name__+'.log'))
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

        self.fmt = "%Y-%m-%d %H:%M:%S"
        self.timezone = 'US/Michigan'

        self.logger.info('{} - MyAI_Model Initialization Finished !'.format(datetime.now(timezone(self.timezone)).strftime(self.fmt)))
    
    
    def process(self, input, context):
        reply_data = proto_sample_pb2.ServerReply()

        try:
            image = _recover_image_from_bytestream(input.img_bytes, input.height, input.width, input.channel)
            self.height, self.width, _ = image.shape
        except Exception as e:
            self.logger.info('{0} - Error Occured: {1}'.format(datetime.now(timezone(self.timezone)).strftime(self.fmt), repr(e)))
            return reply_data

        try:
            def run_model(image):
                image = cv2.flip(image, 1)
                return image
            # run ai model
            self.inference_result = run_model(image)
            
        except Exception as e:
            self.logger.info('{0} - Error Occured: {1}'.format(datetime.now(timezone(self.timezone)).strftime(self.fmt), repr(e)))
            return reply_data
        except KeyboardInterrupt:
            print('==> Finish Model Running.')

        #! test code
        reply_data.img_bytes = bytes(self.inference_result)
        reply_data.height, reply_data.width, reply_data.channel = self.inference_result.shape


        self.logger.info('{} - Reply to request'.format(datetime.now(timezone(self.timezone)).strftime(self.fmt)))
        return reply_data


def opt():
    parser = argparse.ArgumentParser()
    '''------------------------------ gRPC options------------------------------'''
    parser.add_argument('--port', type=str, default='50051', help='Port number, default: 50051')
    parser.add_argument('--num_worker', type=int, default=2, help='the number of threads,. default: 8')
    parser.add_argument('--logdir', type=str, default='./service_log', help='directory where the logging file is saved.')
    return parser.parse_args()

def serve():
    args = opt()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.num_worker))
    proto_sample_pb2_grpc.add_AI_ModelServiceServicer_to_server(
        MyAI_Model(args), server)

    server.add_insecure_port('[::]:%s' % (args.port))
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()