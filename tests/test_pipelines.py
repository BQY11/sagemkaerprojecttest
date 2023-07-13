import pytest
import boto3
import jmespath


@pytest.mark.xfail
def test_that_you_wrote_tests():
    assert False, "No tests written"


def test_pipelines_importable():
    import pipelines  # noqa: F401


# def verify_partial_response(s3_client, s3_list_response, bucket):
#     print('count: ', s3_list_response.get('KeyCount'))
#     keys = jmespath.search('Contents[*].Key', s3_list_response)
#     for key in keys:
#         response = s3_client.head_object(Bucket=bucket, Key=key)
#         print(jmespath.search('ServerSideEncryption', response), jmespath.search('SSEKMSKeyId', response),
#               key.ljust(20)[-20:])
#         assert jmespath.search('ServerSideEncryption', response) == 'aws:kms', "Encryption must be aws:kms"
#
#
# def test_s3_bucket():
#     bucket = 'wnzl.dx.germancredit.model.dev'
#     s3_client = boto3.client('s3')
#     response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=10)
#     verify_partial_response(s3_client, response, bucket)
#     while response.get('NextContinuationToken'):
#         response = s3_client.list_objects_v2(Bucket=bucket, ContinuationToken=response['NextContinuationToken'],
#                                              MaxKeys=10)
#         verify_partial_response(s3_client, response, bucket)
