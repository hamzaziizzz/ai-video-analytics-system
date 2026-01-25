### AI Video Analytics Client Library (Person Detection)

This Python library provides synchronous and asynchronous clients to interact with the AI Video Analytics API.
It focuses on person detection with optional cropped person data in the response.

---

## Features
- **Sync & Async Clients**: `AVASClient` and `AVASClientAsync`
- **Multiple Input Formats**: Image URLs/paths or raw bytes
- **Flexible Configuration**: Thresholds, size filters, limits, response formats
- **MessagePack Support**: Faster binary transfers for large batches
- **Typed Responses**: Pydantic response models

---

## Basic Usage

### Synchronous Client
```python
from ai_video_analytics_clients import AVASClient

client = AVASClient(host="http://localhost", port=18080)
client.server_info()

results = client.extract(
    data=["https://example.com/image.jpg"],
    mode="paths",
    threshold=0.7,
)

print(results["data"][0]["people"])
```

### Asynchronous Client
```python
import asyncio
from ai_video_analytics_clients import AVASClientAsync

async def main():
    client = AVASClientAsync(host="http://localhost", port=18080)
    await client.start()
    try:
        results = await client.extract(
            data=[open("person.jpg", "rb").read()],
            mode="data",
            return_person_data=True,
        )
        print(results["data"][0]["people"])
    finally:
        await client.close()

asyncio.run(main())
```

---

## `extract(...)` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `List[Union[str, bytes]]` | **Required** | Image paths/URLs or raw bytes |
| `mode` | `Literal['paths', 'data']` | `'paths'` | Input type |
| `threshold` | `float` | `0.6` | Confidence threshold |
| `return_person_data` | `bool` | `False` | Include cropped person images |
| `limit_people` | `int` | `0` | Max people per image (0=unlimited) |
| `min_person_size` | `int` | `0` | Minimum person size in pixels |
| `img_req_headers` | `Dict[str, str]` | `None` | Headers for remote image requests |
| `use_msgpack` | `bool` | `True` | Use MessagePack serialization |
| `raw_response` | `bool` | `True` | Return raw dict instead of Pydantic model |

---

## Response Structure

```python
class PeopleResponse(BaseModel):
    num_det: Optional[int]  # 0-based index within the image
    prob: Optional[float]
    bbox: Optional[List[int]]
    class_id: Optional[int]
    class_name: Optional[str]
    track_id: Optional[int]
    persondata: Optional[str]
```

---

## Tips
1. **MessagePack**: Use `use_msgpack=True` for large batches.
2. **Error Handling**: Check `status` field in each image response.
3. **Resource Management**: Always close async client sessions.
