"""
Here we have a subscriber listen for any messages and acknowledge. We also simulating doing work.
"""
import os
import time
import json
import threading
import pandas as pd
from io import BytesIO
from queue import Queue
from google.cloud import storage
from google.cloud import pubsub_v1
from kubernetes import client, config

# Create a pod
def create_pod(job_id):
    pod_name = f"pod-job-{job_id}"
    container_name = f"container-{pod_name}"

    # Define the container
    container = client.V1Container(
        name=container_name,
        image="nginx:latest", 
        ports=[client.V1ContainerPort(container_port=80)]
    )

    pod_spec = client.V1PodSpec(
        containers=[container]  # Pass the container in a list
    )

    pod_metadata = client.V1ObjectMeta(name=pod_name)

    pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=pod_metadata,
        spec=pod_spec
    )

    # Create an instance of the API class to interact with the K8s cluster
    api_response = v1.create_namespaced_pod(
        namespace="default",
        body=pod
    )
    print("Pod created. Status='%s'" % str(api_response.status))

def delete_pod(pod_name):
    v1.delete_namespaced_pod(name=pod_name, namespace="default", body=client.V1DeleteOptions())
    print(f"Pod {pod_name} deleted")

# List all pods
def list_all_pods():
    # Configs can be set in Configuration class directly or using helper utility
    config.load_kube_config("/Users/lixi/.kube/config")

    v1 = client.CoreV1Api()
    print("Listing pods with their IPs:")
    ret = v1.list_pod_for_all_namespaces(watch=False)
    for i in ret.items:
        print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))

# initiate k8s stuff
config.load_kube_config("/Users/lixi/.kube/config")
# instantiate a client for the V1 API group endpoint
v1 = client.CoreV1Api()

# Use a thread-safe queue instead of a list
message_queue = Queue()

def job_scheduler():
    for job_id in message_queue:
        create_pod(job_id)
    
    # put monitoring logic here
    time.sleep(10)

    for job_id in message_queue:
        pod_name = f"pod-job-{job_id}"
        delete_pod(pod_name)

def process_data():
    def get_csv(db, target_csv_file): # Make this a private function
        """
        Loads the csv from the bucket
        """
        try:
            if (db == None):
                raise Exception("No database was given!")
            
            bucket = db.bucket("shiply_csv_bucket")
            blob = bucket.blob(f"data/{target_csv_file}")
            data = blob.download_as_bytes()
            df = pd.read_csv(BytesIO(data))
                
            return df
        except Exception as err:
            print(f"Error fetching csv data!\nError: {err}")
            raise Exception("Fatal Error in fetching csv data!")
        return

    while True:
        # Wait until a message is available
        message_data = message_queue.get()
        print("message_data: ", message_data, "end")
        print(f"[INFO] Processing Data: {message_data}\n")

        name = 123
        create_pod(name)
        delete_pod(f"pod-job-{name}")
        # run job scheduler
        # job_scheduler()

        # test using postman - containerize the pod ... 



        # message_data = json.loads(message_data)
        # target_csv_file = message_data["upload_id"] + "-" + message_data["userID"] + ".csv"

        # client = storage.Client()
        # df = get_csv(client, target_csv_file)

        # print(df)

        # print("Doing some processing...")
        # time.sleep(10)  # Sleep 5 seconds to simulate processing
        # print("Processing Finished.")

        # # Signal that the processing is complete
        # message_queue.task_done()

def callback(message):
    message_queue.put(message.data.decode('utf-8'))
    message.ack()  # Acknowledge the message
    print(f"[INFO] Acknowledged Message Content: {message.data.decode('utf-8')}\n")
    return

def main():
    subscription_id = "test_topic-sub"
    project_id = "special-michelle"

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    
    # Start a separate thread for processing data
    processing_thread = threading.Thread(target=process_data)
    processing_thread.daemon = True  # Daemonize thread
    processing_thread.start()

    # Subscribe and listen for messages
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print("Initialized Application. Listening for messages...")

    try:
        # This will block until an exception is thrown (like KeyboardInterrupt)
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        print("Subscription canceled.")

    print("Exiting application.")

main()
