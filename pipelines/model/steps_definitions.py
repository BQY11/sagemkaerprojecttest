import os

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep,
)

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.transformer import Transformer
from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig,
)
from sagemaker.model_monitor import DatasetFormat
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join

from .params import *


def create_process_step(pipeline_session, role, base_job_prefix, kms_key_id):
    framework_version = "0.23-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type="ml.m5.xlarge",
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/process",
        role=role,
        sagemaker_session=pipeline_session,
        output_kms_key=kms_key_id
    )
    code = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processing.py")
    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="validation", source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(
                output_name="datasets", source="/opt/ml/processing/datasets"
            ),
        ],
        code=code,
        kms_key=kms_key_id
    )
    step_process = ProcessingStep(name="DataProcess", step_args=processor_args)
    return step_process


def create_data_quality_check_step(
        default_bucket,
        base_job_prefix,
        model_package_group_name,
        baseline_dataset,
        check_job_config,
):
    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=baseline_dataset,
        dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
        output_s3_uri=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                base_job_prefix,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "dataqualitycheckstep",
            ],
        ),

    )

    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=skip_check_data_quality,
        register_new_baseline=register_new_baseline_data_quality,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
        model_package_group_name=model_package_group_name,
    )
    return data_quality_check_step


def create_data_bias_check_step(
        default_bucket,
        base_job_prefix,
        model_package_group_name,
        s3_data_input_path,
        check_job_config,
        headers,
        kms_key_id
):
    data_bias_analysis_cfg_output_path = (
        f"s3://{default_bucket}/{base_job_prefix}/databiascheckstep/analysis_cfg"
    )
    data_bias_s3_output_path = Join(
        on="/",
        values=[
            "s3:/",
            default_bucket,
            base_job_prefix,
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            "databiascheckstep",
        ],
    )

    data_bias_data_config = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=data_bias_s3_output_path,
        label="class",
        headers=headers,
        dataset_type="text/csv",
        s3_analysis_config_output_path=data_bias_analysis_cfg_output_path,
    )

    data_bias_config = BiasConfig(
        label_values_or_threshold=[1],
        facet_name="gender",
        facet_values_or_threshold=[0],
        group_name="age",
    )

    data_bias_check_config = DataBiasCheckConfig(
        data_config=data_bias_data_config,
        data_bias_config=data_bias_config,
        kms_key=kms_key_id
    )

    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_data_bias,
        register_new_baseline=register_new_baseline_data_bias,
        supplied_baseline_constraints=supplied_baseline_constraints_data_bias,
        model_package_group_name=model_package_group_name,
    )
    return data_bias_check_step


def create_model_training_step(
        pipeline_session,
        image_uri,
        train_data,
        val_data,
        model_path,
        base_job_prefix,
        role,
        kms_key_id,
):
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/train",
        role=role,
        sagemaker_session=pipeline_session,
        output_kms_key=kms_key_id,
    )

    xgb_train.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        colsample_bytree=0.7,  # 0.94,
        gamma=0.33,
        num_round=40,
        max_depth=3,
        alpha=7,
        eta=0.23,
        subsample=0.7,
    )

    train_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=train_data,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=val_data,
                content_type="text/csv",
            ),
        }
    )
    step_train = TrainingStep(
        name="ModelTraining",
        step_args=train_args,
    )
    return step_train


def create_model_step(pipeline_session, image_uri, role, model_artifact):
    model = Model(
        image_uri=image_uri,
        model_data=model_artifact,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_create_model = ModelStep(
        name="Credit",
        step_args=model.create(
            instance_type="ml.m5.large", accelerator_type="ml.eia1.medium"
        ),
    )

    return step_create_model, model


def create_transform_step(default_bucket, base_job_prefix, model_name, kms_key_id):
    transformer = Transformer(
        model_name=model_name,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        accept="text/csv",
        assemble_with="Line",
        output_path=f"s3://{default_bucket}/{base_job_prefix}/transform",
        output_kms_key=kms_key_id
    )

    step_transform = TransformStep(
        name="OutputTransform",
        transformer=transformer,
        inputs=TransformInput(
            data=test_data,
            input_filter="$[1:]",
            join_source="Input",
            output_filter="$[0,-1]",
            content_type="text/csv",
            split_type="Line",
        ),
    )
    return step_transform


def create_model_quality_check_step(
        default_bucket,
        base_job_prefix,
        transform_output,
        check_job_config,
        model_package_group_name,
):
    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset=transform_output,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                base_job_prefix,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "modelqualitycheckstep",
            ],
        ),
        problem_type="BinaryClassification",
        probability_attribute="_c1",
        ground_truth_attribute="_c0",
        probability_threshold_attribute="0.5",
    )

    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=skip_check_model_quality,
        register_new_baseline=register_new_baseline_model_quality,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
        model_package_group_name=model_package_group_name,
    )

    return model_quality_check_step


def create_model_bias_check_step(
        default_bucket,
        base_job_prefix,
        train_data,
        headers,
        target_label,
        model_name,
        check_job_config,
        model_package_group_name,
        kms_key_id

):
    model_bias_analysis_cfg_output_path = (
        f"s3://{default_bucket}/{base_job_prefix}/modelbiascheckstep/analysis_cfg"
    )

    model_bias_data_config = DataConfig(
        s3_data_input_path=train_data,
        s3_output_path=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                base_job_prefix,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "modelbiascheckstep",
            ],
        ),
        s3_analysis_config_output_path=model_bias_analysis_cfg_output_path,
        label=target_label,
        headers=headers,
        dataset_type="text/csv",
    )

    model_config = ModelConfig(
        model_name=model_name,
        instance_count=1,
        instance_type="ml.m5.xlarge",
    )

    data_bias_config = BiasConfig(
        label_values_or_threshold=[1],
        facet_name="gender",
        facet_values_or_threshold=[0],
        group_name="age",
    )

    model_bias_check_config = ModelBiasCheckConfig(
        data_config=model_bias_data_config,
        data_bias_config=data_bias_config,
        model_config=model_config,
        model_predicted_label_config=ModelPredictedLabelConfig(),
        kms_key=kms_key_id
    )

    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_bias,
        register_new_baseline=register_new_baseline_model_bias,
        supplied_baseline_constraints=supplied_baseline_constraints_model_bias,
        model_package_group_name=model_package_group_name,
    )
    return model_bias_check_step, model_bias_check_config


def create_model_explainability_check_step(
        default_bucket,
        base_job_prefix,
        train_data,
        headers,
        target_label,
        model_name,
        check_job_config,
        model_package_group_name,
        kms_key_id
):
    model_explainability_analysis_cfg_output_path = "s3://{}/{}/{}/{}".format(
        default_bucket, base_job_prefix, "modelexplainabilitycheckstep", "analysis_cfg"
    )

    model_explainability_data_config = DataConfig(
        s3_data_input_path=train_data,
        s3_output_path=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                base_job_prefix,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "modelexplainabilitycheckstep",
            ],
        ),
        s3_analysis_config_output_path=model_explainability_analysis_cfg_output_path,
        label=target_label,
        headers=headers,
        dataset_type="text/csv",
    )

    shap_config = SHAPConfig(
        agg_method="mean_abs",
        save_local_shap_values=True,
        seed=123,
    )

    model_config = ModelConfig(
        model_name=model_name,
        instance_count=1,
        instance_type="ml.m5.xlarge",
    )

    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=model_explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
        kms_key=kms_key_id
    )

    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_explainability,
        register_new_baseline=register_new_baseline_model_explainability,
        supplied_baseline_constraints=supplied_baseline_constraints_model_explainability,
        model_package_group_name=model_package_group_name,
    )
    return model_explainability_check_step, model_explainability_check_config


def create_evaluation_step(
        base_job_prefix, image_uri, test_data, model_artifact, pipeline_session, role, kms_key_id
):
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/evaluation",
        role=role,
        sagemaker_session=pipeline_session,
        output_kms_key=kms_key_id
    )
    code = os.path.join(os.path.dirname(os.path.realpath(__file__)), "evaluation.py")
    eval_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=model_artifact,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=test_data,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
        code=code,
        kms_key=kms_key_id
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step_eval = ProcessingStep(
        name="Evaluation",
        step_args=eval_args,
        property_files=[evaluation_report],
    )
    return step_eval, evaluation_report


def create_model_register_step(
        model, model_metrics, drift_check_baselines, model_package_group_name, step_name
):
    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large", "ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
    )

    step_register = ModelStep(name=step_name, step_args=register_args)
    return step_register
