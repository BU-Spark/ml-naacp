# For production build (Linux based platforms)
docker buildx build --platform linux/amd64 -t us-east1-docker.pkg.dev/special-michelle/naacp/ml_cloud_run .

# For local development (for those with Apple Silicon)
# docker build -t ml_cloud_run .

# We push the image to Michelle's special project
docker push us-east1-docker.pkg.dev/special-michelle/naacp/ml_cloud_run                                   
 

