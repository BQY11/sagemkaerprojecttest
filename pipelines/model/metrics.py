from sagemaker.model_metrics import MetricsSource, ModelMetrics, FileSource
from sagemaker.drift_check_baselines import DriftCheckBaselines


def get_model_metrics(
        data_quality_check_step,
        data_bias_check_step,
        model_quality_check_step,
        model_bias_check_step,
        model_explainability_check_step,
        model_bias_check_config,
        model_explainability_check_config,
):
    model_metrics = ModelMetrics(
        model_data_statistics=MetricsSource(
            s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        bias_pre_training=MetricsSource(
            s3_uri=data_bias_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            s3_uri=model_quality_check_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=model_quality_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),

        explainability=MetricsSource(
            s3_uri=model_explainability_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
            
        bias_post_training=MetricsSource(
            s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        bias=MetricsSource(
            s3_uri='s3://wnzl.dx.germancredit.model.dev/german-credit/gdsmzk7ce7tl/modelbiascheckstep',
            # content_type="application/json",
            # s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
    )

    drift_check_baselines = DriftCheckBaselines(
        model_data_statistics=MetricsSource(
            s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        bias_pre_training_constraints=MetricsSource(
            s3_uri=data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        bias_config_file=FileSource(
            s3_uri=model_bias_check_config.monitoring_analysis_config_uri,
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        bias_post_training_constraints=MetricsSource(
            s3_uri=model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        explainability_constraints=MetricsSource(
            s3_uri=model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        explainability_config_file=FileSource(
            s3_uri=model_explainability_check_config.monitoring_analysis_config_uri,
            content_type="application/json",
        ),
    )

    return model_metrics, drift_check_baselines
