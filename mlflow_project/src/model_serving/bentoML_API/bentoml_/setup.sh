#!/bin/bash
aws ecr get-login-password --profile aws-infrastructure --region eu-central-1 | docker login --username AWS --password-stdin 258781458051.dkr.ecr.eu-central-1.amazonaws.com/saas-ai-insights
