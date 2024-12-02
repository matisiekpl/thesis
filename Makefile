pull:
	aws s3 --region us-east-1 sync s3://mateuszwozniak-thesis-experiments/ .
compile:
	docker build --platform linux/amd64 -t 745637818285.dkr.ecr.eu-central-2.amazonaws.com/thesis:$(VERSION) .
	docker push 745637818285.dkr.ecr.eu-central-2.amazonaws.com/thesis:$(VERSION)
