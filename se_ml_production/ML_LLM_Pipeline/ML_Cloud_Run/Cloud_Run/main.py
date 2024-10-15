import secret
from app_instance import app
from ML_API import ml_router
from global_state import global_instance
from bootstrappers import bootstrap_pipeline, validate_bootstrap, bootstrap_MongoDB_Prod
import logging
from starlette.responses import JSONResponse


# Rate Limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address, default_limits=["3/minute"])

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(status_code=429, content={"message": "Rate limit exceeded"}))
app.add_middleware(SlowAPIMiddleware)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # Enforce HTTPS 
# from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

# app.add_middleware(HTTPSRedirectMiddleware)

# CORS - Cross Origin Resource Sharing

# from fastapi.middleware.cors import CORSMiddleware

# origins = ["webpage"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],  # Limit allowed methods
#     allow_headers=["*"],  # Specify allowed headers
# )

# # IP Whitelisting

from fastapi import HTTPException, Request, Security
from starlette.middleware.base import BaseHTTPMiddleware

WHITELISTED_IPS = ["127.0.0.1", "10.142.15.231", "128.197.28.37", "128.197.28.135"]
class IPWhitelistMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host

        if client_ip not in WHITELISTED_IPS:
            logger.warning(f" Unauthorized IP attempted to connect: {client_ip}")

            # Respond with a custom message
            return JSONResponse(
                status_code=403,
                content={"detail": "Access forbidden: Your IP is not allowed to access this service."}
            )

        response = await call_next(request)
        return response

app.add_middleware(IPWhitelistMiddleware)

app.include_router(ml_router)

@app.on_event("startup")
async def startup_event():

    """
    We store all global variables needed by all functions through FastAPI's app.state
    """
    try:
        # Main pipeline Boostrap
        (db_manager, gcp_db) = bootstrap_pipeline()
        validate_bootstrap(db_manager, gcp_db)
        global_instance.update_data("db_manager", db_manager)
        global_instance.update_data("gcp_db", gcp_db)

        # MongoDB Bootstrap
        defined_collection_names = ["uploads"]
        db_manager = global_instance.get_data("db_manager")
        # We then create our first MongoDB connection
        db_manager.init_connection(uri=secret.MONGO_URI_NAACP)

        db_manager.run_job(
            bootstrap_MongoDB_Prod, 
            db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
            defined_collection_names, # Argument 2
            connection_obj=db_manager.act_con[0]
        )
    except Exception as e:
        print(f"[Error!] FATAL ERROR! | {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # This bootstraps the FastAPI 




















