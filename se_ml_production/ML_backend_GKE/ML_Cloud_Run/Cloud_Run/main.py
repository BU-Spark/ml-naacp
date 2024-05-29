import secret
from app_instance import app
from ML_API import ml_router
from global_state import global_instance
from bootstrappers import bootstrap_pipeline, validate_bootstrap, bootstrap_MongoDB_Prod


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
        defined_collection_names = ["uploads", "discarded"]
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
    print(f"Deployment Test Comlpete with no erors\n")




















