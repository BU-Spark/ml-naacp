from main import app
from bootstrappers import bootstrap_pipeline, validate_bootstrap

@app.on_event("startup")
async def startup_event():
    """
    We store all global variables needed by all functions through FastAPI's app.state
    """
    try:
        (app.state.year,
         app.state.dsource,
         app.state.dname,
         app.state.state,
         app.state.drop_geos,
         app.state.mappings,
         app.state.heir_data,
         app.state.saved_geocodes,
         app.state.nlp_ner,
         app.state.nlp_topic,
         app.state.db) = bootstrap_pipeline()

        validate_bootstrap(
            app.state.year,
            app.state.dsource,
            app.state.dname,
            app.state.state,
            app.state.drop_geos,
            app.state.mappings,
            app.state.heir_data,
            app.state.saved_geocodes,
            app.state.nlp_ner,
            app.state.nlp_topic,
            app.state.db
        )

    except Exception as e:
        print(f"FATAL ERROR! | {e}")
        raise