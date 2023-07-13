import sys
import os

import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_monitor import DatasetFormat
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker import image_uris
from .steps_definitions import *
from .metrics import get_model_metrics
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.functions import Join
from sagemaker import s3
import json
import uuid

#############################
# TODO: delete the following hotfix once Sagemaker SDK fixed the issue
# The sagemaker SDK has a bug in the following function, until it's fixed, we need to fix it like this to support uploading files with explicit KMS encryption
# sagemaker.model_monitor.clarify_model_monitoring.ClarifyModelMonitor._upload_analysis_config
#############################
def hotfix_upload_analysis_config(self, analysis_config, output_s3_uri, job_definition_name):
    """Upload analysis config to s3://<output path>/<job name>/analysis_config.json

    Args:
        analysis_config (dict): analysis config of a Clarify model monitor.
        output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
            Default: "s3://<default_session_bucket>/<job_name>/output"
        job_definition_name (str): Job definition name.
            If not specified then a default one will be generated.

    Returns:
        str: The S3 uri of the uploaded file(s).
    """
    s3_uri = s3.s3_path_join(
        output_s3_uri,
        job_definition_name,
        str(uuid.uuid4()),
        "analysis_config.json",
    )
    # _LOGGER.info("Uploading analysis config to {s3_uri}.")
    return s3.S3Uploader.upload_string_as_file_body(
        json.dumps(analysis_config),
        desired_s3_uri=s3_uri,
        sagemaker_session=self.sagemaker_session,
        # Fix: Specify the KMS key when uploading objects
        kms_key=self.output_kms_key)


sagemaker.model_monitor.clarify_model_monitoring.ClarifyModelMonitor._upload_analysis_config = hotfix_upload_analysis_config


def get_sagemaker_client(region):
    """Gets the sagemaker client.

       Args:
           region: the aws region to start the session
           default_bucket: the bucket to use for storing the artifacts

       Returns:
           `sagemaker.session.Session instance
       """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_pipeline(
        pipeline_name,
        role,
        default_bucket,
        base_job_prefix,
        region,
        kms_key_id,
        model_package_group_name,
        target_label='class',
):
    print(
        f'pipeline_name: {pipeline_name}, role: {role}, model_package_group_name: {model_package_group_name},default_bucket: {default_bucket}')
    pipeline_session = PipelineSession(default_bucket=default_bucket)
    headers = ['class', 'gender', 'checking_status', 'duration', 'credit_history',
               'purpose', 'credit_amount', 'savings_status', 'employment',
               'installment_commitment', 'personal_status', 'other_parties',
               'residence_since', 'property_magnitude', 'age', 'other_payment_plans',
               'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone',
               'foreign_worker']
    process_step = create_process_step(pipeline_session, role, base_job_prefix, kms_key_id)

    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=pipeline_session,
        output_kms_key=kms_key_id
    )

    train_data = process_step.properties.ProcessingOutputConfig.Outputs[
        "train"
    ].S3Output.S3Uri
    val_data = process_step.properties.ProcessingOutputConfig.Outputs[
        "validation"
    ].S3Output.S3Uri

    data_quality_check_step = create_data_quality_check_step(
        default_bucket,
        base_job_prefix,
        model_package_group_name,
        train_data,
        check_job_config,
    )
    data_bias_check_step = create_data_bias_check_step(
        default_bucket,
        base_job_prefix,
        model_package_group_name,
        train_data,
        check_job_config,
        headers,
        kms_key_id
    )

    model_path = f"s3://{default_bucket}/{base_job_prefix}/model"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    train_step = create_model_training_step(
        pipeline_session,
        image_uri,
        train_data,
        val_data,
        model_path,
        base_job_prefix,
        role,
        kms_key_id,
    )

    model_artifact = train_step.properties.ModelArtifacts.S3ModelArtifacts

    model_step, model = create_model_step(
        pipeline_session, image_uri, role, model_artifact
    )

    model_name = model_step.properties.ModelName
    transform_step = create_transform_step(default_bucket, base_job_prefix, model_name, kms_key_id)

    transform_output = transform_step.properties.TransformOutput.S3OutputPath

    model_quality_check_step = create_model_quality_check_step(
        default_bucket,
        base_job_prefix,
        transform_output,
        check_job_config,
        model_package_group_name,
    )
    model_bias_check_step, model_bias_check_config = create_model_bias_check_step(
        default_bucket,
        base_job_prefix,
        train_data,
        headers,
        target_label,
        model_name,
        check_job_config,
        model_package_group_name,
        kms_key_id
    )

    model_explainability_check_step, model_explainability_check_config = create_model_explainability_check_step(
        default_bucket,
        base_job_prefix,
        train_data,
        headers,
        target_label,
        model_name,
        check_job_config,
        model_package_group_name,
        kms_key_id
    )

    eval_step, evaluation_report = create_evaluation_step(
        base_job_prefix, image_uri, test_data, model_artifact, pipeline_session, role, kms_key_id
    )

    model_metrics, drift_check_baselines = get_model_metrics(
        data_quality_check_step,
        data_bias_check_step,
        model_quality_check_step,
        model_bias_check_step,
        model_explainability_check_step,
        model_bias_check_config,
        model_explainability_check_config,
    )

    register_model_step = create_model_register_step(
        model, model_metrics, drift_check_baselines, model_package_group_name, "RegisterModelStaging"
    )
    
    # get inference model group name
    model_package_group_name_temp = model_package_group_name.split("-")
    model_package_group_name_temp[-1] = "inference-" + model_package_group_name_temp[-1]
    model_package_group_inference_name = "-".join(model_package_group_name_temp)
    
    register_model_inference_step = create_model_register_step(
        model, model_metrics, drift_check_baselines, model_package_group_inference_name, "RegisterModelInference"
    )

    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report,
            json_path="classification_metrics.accuracy.value",
        ),
        right=0.7,
    )
    check_accuracy_cond_step = ConditionStep(
        name="CheckAccuracy",
        conditions=[cond_lte],
        if_steps=[register_model_step, register_model_inference_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            instance_type,
            model_approval_status,
            input_data,
            test_data,
            skip_check_data_quality,
            register_new_baseline_data_quality,
            supplied_baseline_statistics_data_quality,
            supplied_baseline_constraints_data_quality,
            skip_check_data_bias,
            register_new_baseline_data_bias,
            supplied_baseline_constraints_data_bias,
            skip_check_model_quality,
            register_new_baseline_model_quality,
            supplied_baseline_statistics_model_quality,
            supplied_baseline_constraints_model_quality,
            skip_check_model_bias,
            register_new_baseline_model_bias,
            supplied_baseline_constraints_model_bias,
            skip_check_model_explainability,
            register_new_baseline_model_explainability,
            supplied_baseline_constraints_model_explainability,
        ],
        steps=[
            process_step,
            data_quality_check_step,
            data_bias_check_step,
            train_step,
            model_step,
            transform_step,
            model_quality_check_step,
            model_bias_check_step,
            model_explainability_check_step,
            eval_step,
            check_accuracy_cond_step
        ],
        sagemaker_session=pipeline_session,
    )
    return pipeline
