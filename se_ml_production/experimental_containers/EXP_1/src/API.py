from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import UploadFile, HTTPException, Query, Body, Form

from typing import Callable
from concurrent import futures
from google.cloud import pubsub_v1

from urllib.parse import unquote_plus

router = APIRouter()

# Some pub/sub funcitons
def get_callback(
	publish_future: pubsub_v1.publisher.futures.Future, data: str
) -> Callable[[pubsub_v1.publisher.futures.Future], None]:
	def callback(publish_future: pubsub_v1.publisher.futures.Future) -> None:
		try:
			# Wait 60 seconds for the publish call to succeed.
			print(publish_future.result(timeout=60))
		except futures.TimeoutError:
			print(f"Publishing {data} timed out.")

	return callback

@router.post("/test_endpoint")
async def upload_RSS(Payload: str = Body(..., description="Some String payload")):
	try:
		decoded_payload = str(unquote_plus(Payload)[8:])
		print("Passed in String:", decoded_payload)

		# Here we send a pub/sub message with our string payload
		project_id = "special-michelle"
		topic_id = "test_topic"

		publisher = pubsub_v1.PublisherClient()
		topic_path = publisher.topic_path(project_id, topic_id)
		publish_futures = []

		data = decoded_payload
		# When you publish a message, the client returns a future.
		publish_future = publisher.publish(topic_path, data.encode("utf-8"))
		# Non-blocking. Publish failures are handled in the callback function.
		publish_future.add_done_callback(get_callback(publish_future, data))
		publish_futures.append(publish_future)

		# Wait for all the publish futures to resolve before exiting.
		futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)

		print(f"Published messages with error handler to {topic_path}.")

		return JSONResponse(content={"String Payload": decoded_payload, "status": "String Successfully Uploaded"}, status_code=200)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")




