import logging
from typing import Union, Dict, Any, List, Literal

import msgpack
import requests
import ujson

from ai_video_analytics_clients.common_utils import decode_people_data, b64_encode_data
from ai_video_analytics_clients.schemas import DetectionResponse

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


class AVASClient:
    """
    Client for interacting with the AI Video Analytics API.

    Provides methods to query server information and perform person detection/extraction.

    Attributes:
       server: Base URL of the AI Video Analytics server
       sess: Persistent HTTP session for connection pooling
    """

    def __init__(self, host: str = 'http://localhost', port: Union[str, int] = 18081) -> None:
        """
        Initialize the AI Video Analytics client.

        Args:
            host: Server hostname or IP address
            port: Server port number
        """
        self.server = f'{host}:{port}'
        self.sess = requests.Session()

    def server_info(self, server: str = None, show: bool = True) -> Dict[str, Any]:
        """
        Retrieve and display server configuration information.

        Args:
            server: Custom server URL (overrides default)
            show: Whether to print formatted server info to console

        Returns:
            Dictionary containing server configuration details
        """
        if server is None:
            server = self.server

        info_uri = f'{server}/info'
        # Execute GET request and parse JSON response
        info = self.sess.get(info_uri).json()

        if show:
            # Extract relevant server information
            server_uri = self.server
            backend_name = info['models']['inference_backend']
            det_name = info['models']['det_name']
            det_batch_size = info['models']['det_batch_size']
            det_max_size = info['models']['max_size']

            # Format and display server information
            print(f'Server: {server_uri}\n'
                  f'    Inference backend:      {backend_name}\n'
                  f'    Detection model:        {det_name}\n'
                  f'    Detection image size:   {det_max_size}\n'
                  f'    Detection batch size:   {det_batch_size}')

        return info

    def extract(
            self,
            data: List[Union[str, bytes]],
            mode: Literal['paths', 'data'] = 'paths',
            threshold: float = 0.6,
            return_person_data: bool = False,
            limit_people: int = 0,
            min_person_size: int = 0,
            img_req_headers: Dict[str, str] = None,
            use_msgpack: bool = True,
            raw_response: bool = True
    ) -> Union[DetectionResponse, dict]:

        """
        Perform person extraction on input data.

        Supports:
        - Image URI
        - Raw image bytes
        - Base64-encoded images

        Args:
            data: List of image paths (mode='paths') or image bytes (mode='data')
            mode: Input type - 'paths' for image URLs/paths, 'data' for binary images
            threshold: Confidence threshold for person detection (0.0-1.0)
            return_person_data: Whether to include decoded person images in response
            limit_people: Maximum people to process per image (0 = no limit)
            min_person_size: Persons smaller than this value will be ignored (0 = no limit)
            img_req_headers: Headers to use for requesting images from remote servers.
            use_msgpack: Use MessagePack for faster binary serialization and bandwidth savings.
            raw_response: Return raw dictionary instead of parsed response object

        Returns:
            Processed detection results either as DetectionResponse object or raw dict
        """

        if not img_req_headers:
            img_req_headers = {}

        extract_uri = f'{self.server}/detect'

        # Prepare image data based on input mode
        images: Dict[str, Any]
        if mode == 'data':
            if not use_msgpack:
                # Convert binary data to base64 strings if not using msgpack
                images = dict(data=b64_encode_data(data))
            else:
                images = dict(data=data)
        elif mode == 'paths':
            images = dict(urls=data)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'paths' or 'data'")

        # Build request payload
        req = dict(images=images,
                   threshold=threshold,
                   return_person_data=return_person_data,
                   limit_people=limit_people,  # 0 = process all detected people
                   min_person_size=min_person_size,
                   msgpack=use_msgpack,
                   img_req_headers=img_req_headers
                   )

        # Send request with appropriate serialization
        if use_msgpack:
            # MessagePack binary format
            resp = self.sess.post(
                extract_uri,
                data=msgpack.dumps(req),
                timeout=120,
                headers={
                    'content-type': 'application/msgpack',
                    'accept': 'application/x-msgpack'
                }
            )
        else:
            # Standard JSON format
            resp = self.sess.post(extract_uri, json=req, timeout=120)

        # Parse response based on content type
        if resp.headers['content-type'] == 'application/x-msgpack':
            content = msgpack.loads(resp.content)
        else:
            content = ujson.loads(resp.content)

        # Decode person images if requested
        if return_person_data:
            content = decode_people_data(content)

        # Return either raw dict or validated Pydantic model
        return content if raw_response else DetectionResponse.model_validate(content)
