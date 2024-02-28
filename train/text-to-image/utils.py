def create_bucket_if_not_exists(training_bucket):
    import boto3
    import botocore

    mySession = boto3.session.Session()
    AwsRegion = mySession.region_name



    s3_rsrc = boto3.resource('s3')

    try:
        s3_rsrc.meta.client.head_bucket(Bucket=training_bucket)
        print(f'Using an existing bucket {training_bucket}')
    except botocore.exceptions.ClientError as e:
        if e.response.get('Error') is not None and e.response['Error'].get('Code') is not None and e.response['Error'][
            'Code'] == '404':
            try:
                if AwsRegion == 'us-east-1':
                    s3_bucket = s3_rsrc.create_bucket(Bucket=training_bucket)
                    print(f'Created {s3_bucket}')
                else:
                    s3_bucket = s3_rsrc.create_bucket(Bucket=training_bucket,
                                                      CreateBucketConfiguration={'LocationConstraint': AwsRegion})
                    print(f'Created {s3_bucket}')
            except Exception as e:
                raise Exception(f'Error Creating S3 bucket {training_bucket}.\n' + str(e))
        else:
            raise Exception(f'Error checking if the bucket {training_bucket} exists or not.\n' + str(e))
    except Exception as e:
        raise Exception(f'Error checking if the bucket {training_bucket} exists or not. \n' + str(e))



def wait_dataset_creation_cl(rekognition, dataset):
    import time

    ## Wait for the creation of the dataset to complete
    chk_status = True
    starttime = time.time()
    while chk_status:
        ## wait for 15 seconds. To check status every 15 seconds
        time.sleep(15)
        dataset_status = rekognition.describe_dataset(DatasetArn=dataset["DatasetArn"])
        if dataset_status["DatasetDescription"]["Status"] != "CREATE_IN_PROGRESS":
            chk_status = False
        ## Continue to check for status for 1 hour
        if (time.time() - starttime) > 3600:
            chk_status = False
            # Raise an exception
            raise Exception("Error creating importing Dataset")
    return dataset_status

def wait_cl_model_stop(rekognition, cl_project, model_version_name):
    import time
    ## Wait for the model to stop
    chk_status = True
    starttime = time.time()
    while chk_status:
        ## wait for 1 minute. To check status every 1 minute
        time.sleep(60)
        model_stop_status = rekognition.describe_project_versions(
            ProjectArn=cl_project["ProjectArn"], VersionNames=[model_version_name]
        )
        if model_stop_status["ProjectVersionDescriptions"][0]["Status"] != "STOPPING":
            chk_status = False
        ## Continue to check for status for 1 hour
        if (time.time() - starttime) > 3600:
            chk_status = False
            # Raise an exception
            raise Exception("Error Creating Stopping a Rekognition Custom labels Model")
    return model_stop_status

