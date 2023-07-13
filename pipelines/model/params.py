from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
    ParameterBoolean,
)

# inputs
input_data = ParameterString(name="InputData")
test_data = ParameterString(name="TestData")

# instaces
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")

# approval params
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
)
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.5)

# for data quality check step
skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=False)
register_new_baseline_data_quality = ParameterBoolean(
    name="RegisterNewDataQualityBaseline", default_value=False
)
supplied_baseline_statistics_data_quality = ParameterString(
    name="DataQualitySuppliedStatistics", default_value=""
)
supplied_baseline_constraints_data_quality = ParameterString(
    name="DataQualitySuppliedConstraints", default_value=""
)

# for data bias check step
skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value=False)
register_new_baseline_data_bias = ParameterBoolean(
    name="RegisterNewDataBiasBaseline", default_value=False
)
supplied_baseline_constraints_data_bias = ParameterString(
    name="DataBiasSuppliedBaselineConstraints", default_value=""
)

# for model quality check step
skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value=False)
register_new_baseline_model_quality = ParameterBoolean(
    name="RegisterNewModelQualityBaseline", default_value=False
)
supplied_baseline_statistics_model_quality = ParameterString(
    name="ModelQualitySuppliedStatistics", default_value=""
)
supplied_baseline_constraints_model_quality = ParameterString(
    name="ModelQualitySuppliedConstraints", default_value=""
)

# for model bias check step
skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=False)
register_new_baseline_model_bias = ParameterBoolean(
    name="RegisterNewModelBiasBaseline", default_value=False
)
supplied_baseline_constraints_model_bias = ParameterString(
    name="ModelBiasSuppliedBaselineConstraints", default_value=""
)

# for model explainability check step
skip_check_model_explainability = ParameterBoolean(
    name="SkipModelExplainabilityCheck", default_value=False
)
register_new_baseline_model_explainability = ParameterBoolean(
    name="RegisterNewModelExplainabilityBaseline", default_value=True
)
supplied_baseline_constraints_model_explainability = ParameterString(
    name="ModelExplainabilitySuppliedBaselineConstraints", default_value=""
)
