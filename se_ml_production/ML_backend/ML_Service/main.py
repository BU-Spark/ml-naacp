from app_instance import app
from ML_API import ml_router
from events import startup_event

startup_event() # ML Bootstrapper, comment this if you just want to test FastAPI standalone or csv_funcs
app.include_router(ml_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # This bootstraps the FastAPI 




















