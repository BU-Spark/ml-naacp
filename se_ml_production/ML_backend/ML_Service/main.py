import nltk
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
        db) = bootstrap_pipeline()

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
            db
        )

        nltk.download('punkt')

        # MongoDB Bootstrap
        defined_collection_names = ["uploads", "discarded"]
        db_prod = connect_MongoDB_Prod()
        global_instance.update_data("db_prod", db_prod)
        bootstrap_MongoDB_Prod(db_prod, defined_collection_names)

    except Exception as e:
        print(f"[Error!] FATAL ERROR! | {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # This bootstraps the FastAPI 




















