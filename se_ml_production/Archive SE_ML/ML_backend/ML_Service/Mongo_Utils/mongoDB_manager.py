import uuid
import time
from pymongo import MongoClient 

from pymongo.errors import InvalidOperation, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError

class MongoDBManager:
    """
    A MongoDB Manager. This avoids all the loose MongoDB connections and consoldiates them in one place.

    Note: Its resource intensive to instantiate a MongoDB connection
    Note: I didn't create one uri, because what if you want to connect to multiple MongoDB DBs?
    """
    def __init__(self):
        """
        MongoDB Manager Initalizer
        ---------------------------
        uri: The MongoDB URI
        act_con: Active Connections
        init_connection: This creates the client object
        hist_connections_closed: Number of times a connection was closed.
        hist_connections_opened: Number of times a connection was opened.
        jobs_count: Number of jobs passed to MongoDB Manager
        """
        self.act_con = []
        self.upper_bound_con = 50 # We limit ourselves to 50 cocurrent conneciton
        self.hist_connections_closed = 0
        self.hist_connections_opened = 0
        self.jobs_count = 0
        self.job_queue = [] # For async purposes in the future...
        
    def __str__(self):
        """
        Some useful metrics.
        """
        connection_list = []
        for connection in self.act_con:
            try:
                connection_list.append({connection['id']: self.ping(connection['connection'])})
            except InvalidOperation as InvOp: # For any bad connections we see...
                connection_list.append({connection['id']: "BAD CONNECTION"})
                continue
        return(
        f"========================\n"
        f"MongoDB Manager 1.0\n"
        f"- Avaliable Metrics -\n"
        f"========================\n"
        f"Connections: \n{connection_list} \n\n"
        f"Number of Active Connections: {len(self.act_con)}\n"
        f"Historical Number of Connections Closed: {self.hist_connections_closed}\n"
        f"Historical Number of Open Connections: {self.hist_connections_opened}\n"
        f"Number of jobs processed: {self.jobs_count}\n"
        )

    def init_connection(self, uri):
        """
        Attempts to initalize the connection. Should Instantiate a MongoDB Client object.
        """
        try:
            if (len(self.act_con) >= self.upper_bound_con):
                # We might run into this error if we parallize our containers that need each MongoDB manager
                raise Exception("Cannot create anymore connections! Aborting...")
            client = MongoClient(uri)
            unique_id = uuid.uuid4()
            self.act_con.append({
                'id': str(unique_id), 
                'connection': client,
                'uri': uri
            })
            self.hist_connections_opened += 1
            return
        except Exception as e:
            print(f"Original Error: {e}\n\n")
            raise Exception("[FATAL ERROR!] MongoDB Manager failed to initalize a MongoDB connection. Aborting...")
        return

    def force_close_connection(self, unique_id=None):
        """
        Attempts to force close/close the MongoDB connection.

        If unique_id is not provided, then the first connection is closed.
        """
        try:
            if (len(self.act_con) < 1):
                print("[Warning] No connections to close! Aborting...")
                return
                
            connection_obj = None
            
            if (unique_id):
                for connection in self.act_con:
                    if (connection['id'] == unique_id):
                        connection_obj = connection
                        break

                if (not connection_obj):
                    raise Exception("[FATAL ERROR!] MongoDB Manager failed find the specified connection. Aborting...")
            else:
                connection_obj = self.act_con[0] # Close the first connection in the list

            connection_obj['connection'].admin.command('ping') # We need to ping if the connection is alive
            connection_obj['connection'].close();
            print(f"[INFO] Connection was successfully forcefully closed at id: {connection_obj['id']}.")
            self.hist_connections_closed += 1

            if (unique_id):
                # Remove the connection from the list of active connections
                self.act_con = [d for d in self.act_con if d['id'] != unique_id]
            else:
                del self.act_con[0] # Remove the first connection in the list
                
            print(f"[INFO] Connection was successfully removed from active connection list.")
            return
        except InvalidOperation as InvOp:
            print(f"[Warning] Original Error: {InvOp}")
            print("[Warning] Tried to close an already closed connection!")
        except Exception as e:
            print(f"Original Error: {e}\n\n")
            raise Exception("[FATAL ERROR!] MongoDB Manager failed close a MongoDB connection. Aborting...")
        return
                            
            
    def run_job(self, func, *args, connection_obj=None, retry_times=10, retry_time=5, db="se_naacp_gbh", **kwargs):
        """
        A wrapper for all MongoDB jobs. We can retry a "x" amount of times 
        -----
        *** IMPORTANT ***
        PLEASE HAVE THE CLIENT OR CONNECTION BE THE FIRST ARGUMENT OF YOUR PYTHON FUNCTION
        retry_times: The number of times to retry for the connects
        connection: The PyMongo Connection you want to handle this job
        func: The function you want to execute
        """
        try:
            new_args = None
            
            if(not connection_obj): # If no connection is provided, we take the first one
                connection_obj = self.act_con[0]
            
            for retries in range(1, retry_times+2):
                try:    
                    print("[INFO] Checking MongoDB Connection...")
                    if (not self.ping(connection_obj['connection']) == {'ok': 1.0}): # If MongoDB doesn't respond, then we need to force a connection
                        raise InvalidOperation("Connection ping failed.")
                    else:
                        # Perform a simple operation to check if the connection is closed
                        db_test = connection_obj['connection'][db]
                        db_test.list_collection_names()
                        break
                except (Exception, InvalidOperation, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError) as BADCONN:
                    if (retries == retry_times+1):
                        print(f"Original Error: {BADCONN}\n\n")
                        raise Exception("[FATAL ERROR!] MongoDB Manager Exhausted all retries! Aborting...")
                    try:   
                        print(f"[Warning] Failed MongoDB Connection! Retrying in {retry_time} seconds {retries}/{retry_times}...")
                        time.sleep(retry_time)
                        print("[1/3] New Initializing Connection...")
                        self.init_connection(uri=connection_obj['uri'])
                        print("[2/3] Close the current connection...")
                        self.force_close_connection(unique_id=connection_obj['id'])
                        print("[3/3] Changed connection object...")
                        connection_obj = self.act_con[len(self.act_con) - 1]
                        new_args = list(args) # Create new args by copying old args
                        new_args[0] = connection_obj['connection']
                        new_args = tuple(new_args) # Change it back into a tuple
                    except (Exception, InvalidOperation, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError) as e:
                        print(f"[Warning] THIS Retry failed as {e}") # We print and do nothing...
                    continue
                    
            # == The function ==
            if (new_args):
                result = func(*new_args, **kwargs) # This is the same function with a swapped DB link
            else:  
                result = func(*args, **kwargs)
            self.jobs_count += 1
            
            return result # Return anything if needed
        except Exception as e:
            print(f"Original Error: {e}\n\n")
            raise Exception("[FATAL ERROR!] MongoDB Manager failed to run a job. Aborting...")
        return
        
    ### Some Utility functions ###
    def ping(self, connection):
        """
        Pings the MongoDB server.
        """
        return connection.admin.command('ping')

    def remove_connection_from_active_list(unique_id):
        """
        Removes a connection id forcefully from the list. (This should only be used if there is a bug!)
        """
        self.act_con = [d for d in self.act_con if d['id'] != unique_id]
        return