pip install awscli
scp -r ~/.aws user@216.153.52.140:/home/user/.aws
aws s3 cp --endpoint-url=https://99e0307e13e8a8b0820c9c383003db99.r2.cloudflarestorage.com model-file.tar.gz s3://storage-nafkhanzam-com/thesis/models/
