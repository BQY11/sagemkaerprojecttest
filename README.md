## Layout of the Model Build Project

This template project contains all the configuration files and python scripts to build a model based on german credit dataset on Data Science Workbench platform.
The following section provides an overview of how the code is organized and what you need to modify. 

```
|-- pipelines
|   |-- model
|   |   |-- __init__.py
|   |   |-- evaluation.py
|   |   |-- metrics.py
|   |   |-- params.py
|   |   |-- pipeline.py
|   |   |-- processing.py
|   |   |-- steps_definitions.py
|   |-- __init__.py
|   |-- __version__.py
|   |-- _utils.py
|   |-- get_pipeline_definition.py
|   |-- run_pipeline.py
|-- tests
|   -- test_pipelines.py
|-- buildspec.yml
|-- CONTRIBUTING.md
|-- README.md
|-- setup.cfg
|-- setup.py
|-- tox.ini
```

### Files to Customize for your own Model
The following files need be customized for your own model
```
|-- pipelines
|   |-- model
|   |   |-- evaluation.py
|   |   |-- metrics.py
|   |   |-- params.py
|   |   |-- pipeline.py
|   |   |-- processing.py
|   |   |-- steps_definitions.py
|   |-- run_pipeline.py
|-- tests
|   -- test_pipelines.py
```
#### run_pipeline.py
You need to specify your input data here, such as training and testing data.
#### steps_definitions.py
This script defines all the Sagemaker pipeline steps
#### pipeline.py 
It contains the core logic to build the model. The sagemaker pipeline steps are defined here.
#### params.py and metrics.py
They defined the pipeline parameters and model metrics
#### processing.py
This script is for processing step
#### evalutaiton.py
This script is used for evaluation step
#### test_pipelines.py
In case you need run some tests for your model

## Start here
The following AWS CodeBuild specification file defines how the Sagemaker Pipeline is created and run in AWS platform.
The following fields can be customized only:
    `SAGEMAKER_JOB_PREFIX` and `--tags`, all other environment variables are defined and provided by Data Science Workbench platform.

1. ENVIRONMENT - either DEV or PROD
2. AWS_REGION - the current region of runtime, always be ap-southeast-2
3. SAGEMAKER_PIPELINE_ROLE_ARN - the AWS role to run the Sagemaker pipeline, come to Data Science Workbench team to ask for more permissions if needed
4. SAGEMAKER_MODEL_GROUP_NAME - the given model group name
5. SAGEMAKER_PIPELINE_NAME - the given pipeline name
6. ARTIFACT_BUCKET - the given S3 bucket
7. KMS_KEY_ID - the KMS key id used to encrypt above S3 bucket

```
|-- buildspec.yml
```
 
 
