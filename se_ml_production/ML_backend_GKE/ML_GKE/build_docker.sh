# Build for production on Linux Environments
docker buildx build --platform linux/amd64 -t us-east1-docker.pkg.dev/special-michelle/naacp/ml_service_gke .

# We push the image to Michelle's special project
docker push us-east1-docker.pkg.dev/special-michelle/naacp/ml_service_gke                                    

