import nltk
import secret
from app_instance import app
from ML_API import ml_router
from global_state import global_instance
from Mongo_Utils.mongo_funcs import connect_MongoDB_Prod
from bootstrappers import bootstrap_pipeline, validate_bootstrap, bootstrap_MongoDB_Prod

# from events import startup_event
#startup_event() # ML Bootstrapper, comment this if you just want to test FastAPI standalone or csv_funcs

app.include_router(ml_router)

@app.on_event("startup")
async def startup_event():

    """
    We store all global variables needed by all functions through FastAPI's app.state
    """
    try:
        # Main pipeline Boostrap
        (year,
        dsource,
        dname,
        state,
        drop_geos,
        mappings,
        census_base,
        heir_data,
        saved_geocodes,
        nlp_ner,
        nlp_topic,
        db,
        db_manager) = bootstrap_pipeline()

        global_instance.update_data("year", year)
        global_instance.update_data("dsource", dsource)
        global_instance.update_data("dname", dname)
        global_instance.update_data("state", state)
        global_instance.update_data("drop_geos", drop_geos)
        global_instance.update_data("mappings", mappings)
        global_instance.update_data("census_base", census_base)
        global_instance.update_data("heir_data", heir_data)
        global_instance.update_data("saved_geocodes", saved_geocodes)
        global_instance.update_data("nlp_ner", nlp_ner)
        global_instance.update_data("nlp_topic", nlp_topic)
        global_instance.update_data("db_manager", db_manager)

        validate_bootstrap(
            year,
            dsource,
            dname,
            state,
            drop_geos,
            mappings,
            census_base,
            heir_data,
            saved_geocodes,
            nlp_ner,
            nlp_topic,
            db,
            db_manager
        )

        nltk.download('punkt')

        # MongoDB Bootstrap
        defined_collection_names = ["uploads", "discarded"]
        db_prod = connect_MongoDB_Prod()
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




















